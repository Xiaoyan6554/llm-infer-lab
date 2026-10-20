#include "kv_cache_manager.h"

#include <stdexcept>

namespace kvc {

// ---------------- BlockAllocator ----------------

BlockAllocator::BlockAllocator(int num_blocks) : refcount_(num_blocks, 0) {
  if (num_blocks <= 0) throw std::invalid_argument("num_blocks must be > 0");
  free_list_.reserve(num_blocks);
  // 逆序压栈，这样最先分配出去的是 0 号块，便于调试时看 block table。
  for (int i = num_blocks - 1; i >= 0; --i) free_list_.push_back(i);
}

std::optional<BlockId> BlockAllocator::allocate() {
  if (free_list_.empty()) return std::nullopt;
  BlockId id = free_list_.back();
  free_list_.pop_back();
  refcount_[id] = 1;
  return id;
}

void BlockAllocator::incref(BlockId id) {
  if (refcount_.at(id) <= 0) throw std::logic_error("incref on a free block");
  ++refcount_[id];
}

bool BlockAllocator::decref(BlockId id) {
  if (refcount_.at(id) <= 0) throw std::logic_error("double free");
  if (--refcount_[id] == 0) {
    free_list_.push_back(id);
    return true;
  }
  return false;
}

bool BlockAllocator::decref_retain(BlockId id) {
  if (refcount_.at(id) <= 0) throw std::logic_error("double free");
  return --refcount_[id] == 0;
}

void BlockAllocator::revive(BlockId id) {
  if (refcount_.at(id) != 0) throw std::logic_error("revive on a block that is in use");
  refcount_[id] = 1;
}

// ---------------- KVCacheManager ----------------

KVCacheManager::KVCacheManager(int num_blocks, int block_size, bool enable_prefix_caching)
    : KVCacheManager(KVCacheConfig{num_blocks, block_size, enable_prefix_caching, false, 0.0}) {}

KVCacheManager::KVCacheManager(const KVCacheConfig& cfg)
    : block_size_(cfg.block_size),
      enable_prefix_caching_(cfg.enable_prefix_caching),
      lru_eviction_(cfg.lru_eviction),
      watermark_blocks_(static_cast<int>(cfg.watermark * cfg.num_blocks)),
      allocator_(cfg.num_blocks),
      block_hash_(cfg.num_blocks),
      block_fill_(cfg.num_blocks, 0),
      lru_pos_(cfg.num_blocks),
      in_lru_(cfg.num_blocks, 0) {
  if (cfg.block_size <= 0) throw std::invalid_argument("block_size must be > 0");
  if (cfg.watermark < 0.0 || cfg.watermark >= 1.0) throw std::invalid_argument("watermark must be in [0, 1)");
}

// FNV-1a 风格的 64 位哈希，把"前一个块的哈希"也混进来，形成哈希链：
// 只有从第一个块开始完全相同的前缀，才会得到相同的哈希。
uint64_t KVCacheManager::hash_block(uint64_t prev_hash, const Token* begin, int n) {
  uint64_t h = 1469598103934665603ULL ^ prev_hash;
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<uint32_t>(begin[i]);
    h *= 1099511628211ULL;
    h ^= h >> 29;
  }
  return h;
}

std::optional<BlockId> KVCacheManager::allocate_block() {
  auto id = allocator_.allocate();
  if (!id && !lru_.empty()) {
    // 没有真正空闲的块了：驱逐 LRU 队头（最久没被用过的缓存块），把它当新块用。
    BlockId victim = lru_.front();
    lru_.pop_front();
    in_lru_[victim] = 0;
    unregister_block(victim);
    allocator_.revive(victim);
    ++evicted_blocks_;
    id = victim;
  }
  if (id) {
    block_fill_[*id] = 0;
    block_hash_[*id].reset();
  }
  return id;
}

void KVCacheManager::unregister_block(BlockId id) {
  if (!block_hash_[id]) return;
  auto it = cache_.find(*block_hash_[id]);
  if (it != cache_.end() && it->second == id) cache_.erase(it);
  block_hash_[id].reset();
}

void KVCacheManager::release_block(BlockId id) {
  if (lru_eviction_ && block_hash_[id]) {
    // 登记过前缀缓存的满块：引用计数归零后放进 LRU 队尾，内容和缓存登记都保留，之后还能被命中。
    if (allocator_.decref_retain(id)) {
      lru_pos_[id] = lru_.insert(lru_.end(), id);
      in_lru_[id] = 1;
    }
    return;
  }
  if (allocator_.decref(id)) {
    // 块真正被释放：从前缀缓存里摘掉。
    unregister_block(id);
    block_fill_[id] = 0;
  }
}

void KVCacheManager::acquire_cached_block(BlockId id) {
  if (in_lru_[id]) {
    lru_.erase(lru_pos_[id]);
    in_lru_[id] = 0;
    allocator_.revive(id);
  } else {
    allocator_.incref(id);
  }
}

void KVCacheManager::maybe_register_full_block(Sequence& s, int logical_idx) {
  uint64_t prev = logical_idx == 0 ? 0 : s.hashes[logical_idx - 1];
  uint64_t h = hash_block(prev, s.tokens.data() + logical_idx * block_size_, block_size_);
  s.hashes.push_back(h);
  if (!enable_prefix_caching_) return;
  BlockId id = s.blocks[logical_idx];
  if (cache_.emplace(h, id).second) block_hash_[id] = h;
}

bool KVCacheManager::add_sequence(SeqId seq, const std::vector<Token>& prompt) {
  if (seqs_.count(seq)) throw std::invalid_argument("sequence already exists");
  if (prompt.empty()) throw std::invalid_argument("prompt must not be empty");

  const int n = static_cast<int>(prompt.size());
  const int num_full = n / block_size_;
  const int num_blocks = (n + block_size_ - 1) / block_size_;

  // 第一步：只读地算出能命中多少个前缀块（必须从第 0 块开始连续命中）。
  std::vector<uint64_t> hashes;
  std::vector<BlockId> hits;
  int hits_in_lru = 0;
  uint64_t prev = 0;
  for (int i = 0; i < num_full; ++i) {
    uint64_t h = hash_block(prev, prompt.data() + i * block_size_, block_size_);
    hashes.push_back(h);
    prev = h;
    if (enable_prefix_caching_ && hits.size() == static_cast<size_t>(i)) {
      auto it = cache_.find(h);
      if (it != cache_.end()) {
        hits.push_back(it->second);
        if (in_lru_[it->second]) ++hits_in_lru;
      }
    }
  }

  // 第二步：块不够就直接失败，保证"失败不改状态"。
  // 命中的 LRU 块会被复活，不能再算作可驱逐的空间。
  const int need_new = num_blocks - static_cast<int>(hits.size());
  const int available = num_free_blocks() - hits_in_lru;
  if (need_new > available) return false;
  // 水位线：已经有序列在跑时，放进新请求后至少要留 watermark_blocks_ 个可用块给它们 decode。
  if (!seqs_.empty() && available - need_new < watermark_blocks_) return false;

  // 第三步：真正分配。先复活/引用命中的块，再分配新块（避免把刚命中的块驱逐掉）。
  Sequence s;
  s.tokens = prompt;
  s.hashes = std::move(hashes);
  for (BlockId id : hits) {
    acquire_cached_block(id);
    s.blocks.push_back(id);
  }
  prefix_hit_blocks_ += static_cast<int64_t>(hits.size());

  for (int i = static_cast<int>(hits.size()); i < num_blocks; ++i) {
    BlockId id = *allocate_block();  // 前面检查过数量，这里一定成功
    s.blocks.push_back(id);
    block_fill_[id] = (i < num_full) ? block_size_ : n - num_full * block_size_;
    if (i < num_full && enable_prefix_caching_) {
      if (cache_.emplace(s.hashes[i], id).second) block_hash_[id] = s.hashes[i];
    }
  }
  seqs_.emplace(seq, std::move(s));
  return true;
}

bool KVCacheManager::append_token(SeqId seq, Token tok) {
  Sequence& s = seqs_.at(seq);
  const int pos = static_cast<int>(s.tokens.size());  // 新 token 的位置
  const int offset = pos % block_size_;

  if (offset == 0) {
    // 上一个块已写满（或者还没有块），需要一个新块。
    auto id = allocate_block();
    if (!id) return false;
    s.blocks.push_back(*id);
  } else {
    BlockId last = s.blocks.back();
    if (allocator_.refcount(last) > 1) {
      // 最后一个块被别的序列共享：写时复制。
      auto dst = allocate_block();
      if (!dst) return false;
      pending_copies_.push_back({last, *dst});
      block_fill_[*dst] = block_fill_[last];
      release_block(last);
      s.blocks.back() = *dst;
    }
  }

  s.tokens.push_back(tok);
  BlockId cur = s.blocks.back();
  block_fill_[cur] = offset + 1;
  if (offset + 1 == block_size_) {
    maybe_register_full_block(s, static_cast<int>(s.blocks.size()) - 1);
  }
  return true;
}

void KVCacheManager::fork(SeqId parent, SeqId child) {
  if (seqs_.count(child)) throw std::invalid_argument("child sequence already exists");
  const Sequence& p = seqs_.at(parent);
  for (BlockId id : p.blocks) allocator_.incref(id);
  seqs_.emplace(child, p);  // 复制的只是 block table 和 token 列表，物理块是共享的
}

void KVCacheManager::free_sequence(SeqId seq) {
  auto it = seqs_.find(seq);
  if (it == seqs_.end()) return;
  // 逆序释放：尾部的块先回到空闲栈（或先进 LRU，于是先被驱逐），前缀块后回去。
  for (auto b = it->second.blocks.rbegin(); b != it->second.blocks.rend(); ++b) release_block(*b);
  seqs_.erase(it);
}

const std::vector<BlockId>& KVCacheManager::block_table(SeqId seq) const {
  return seqs_.at(seq).blocks;
}

int KVCacheManager::num_tokens(SeqId seq) const {
  return static_cast<int>(seqs_.at(seq).tokens.size());
}

std::vector<CopyOp> KVCacheManager::take_copy_ops() {
  std::vector<CopyOp> out;
  out.swap(pending_copies_);
  return out;
}

double KVCacheManager::slot_utilization() const {
  int used_blocks = num_used_blocks();
  if (used_blocks == 0) return 0.0;
  int64_t filled = 0;
  for (int id = 0; id < allocator_.num_total(); ++id) {
    if (allocator_.refcount(id) > 0) filled += block_fill_[id];
  }
  return static_cast<double>(filled) / (static_cast<double>(used_blocks) * block_size_);
}

}  // namespace kvc
