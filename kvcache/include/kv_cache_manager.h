// 分页 KV Cache 管理器：PagedAttention 显存管理思路的 CPU 版实现。
//
// 核心思想：
//   1. 把 KV cache 显存切成固定大小的物理块（block），每块存 block_size 个 token 的 K/V。
//   2. 每个序列维护一张 block table：逻辑块号 -> 物理块号，物理块不需要连续。
//   3. 按需分配：序列每长出 block_size 个 token 才申请一个新块，浪费最多 block_size-1 个槽位。
//   4. 引用计数 + 写时复制（Copy-on-Write）：fork 出的子序列共享父序列的块，谁先写谁复制。
//   5. 前缀缓存（Prefix Caching）：写满的块按"前缀哈希链"登记，内容相同的前缀直接复用物理块。
//   6. LRU 延迟驱逐（可选）：引用计数归零的缓存块先不失效，空闲块用完时才驱逐最久没用的。
//   7. 准入水位线（可选）：新请求分配完后至少留一部分空闲块给正在 decode 的序列，减少抢占。
//
// 这里不存真正的 K/V 张量，只管理"哪个 token 存在哪个物理块"，
// 这正是 vLLM 调度器里 BlockManager 负责的部分。

#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kvc {

using BlockId = int32_t;
using SeqId = int64_t;
using Token = int32_t;

// 物理块分配器：固定数量的块 + 空闲栈 + 引用计数。
class BlockAllocator {
 public:
  explicit BlockAllocator(int num_blocks);

  // 取一个空闲块，引用计数置为 1；没有空闲块时返回 std::nullopt。
  std::optional<BlockId> allocate();
  void incref(BlockId id);
  // 引用计数减一；返回 true 表示归零、块已回到空闲池。
  bool decref(BlockId id);
  // 引用计数减一，但归零时不放回空闲栈：块的内容还要留着给前缀缓存命中，由调用方（LRU）管理。
  bool decref_retain(BlockId id);
  // 让一个引用计数为 0、又不在空闲栈里的块重新被持有（LRU 里的块被命中，或被驱逐后复用）。
  void revive(BlockId id);

  int refcount(BlockId id) const { return refcount_.at(id); }
  int num_free() const { return static_cast<int>(free_list_.size()); }
  int num_total() const { return static_cast<int>(refcount_.size()); }

 private:
  std::vector<int> refcount_;
  std::vector<BlockId> free_list_;  // 当作栈用：后释放的先被复用
};

// 一次写时复制：执行层需要把 src 块的 K/V 拷贝到 dst 块（在 GPU 上是一次 memcpy）。
struct CopyOp {
  BlockId src;
  BlockId dst;
};

struct KVCacheConfig {
  int num_blocks = 0;
  int block_size = 16;
  bool enable_prefix_caching = false;
  // 开启后，引用计数归零的缓存块进入 LRU 队列，仍然可以被新请求命中；
  // 只有在没有真正空闲的块时，才驱逐最久没被用过的那个。需要同时开启前缀缓存才有意义。
  bool lru_eviction = false;
  // 准入水位线（占总块数的比例）：add_sequence 分配完后，可用块不能低于这个比例，
  // 否则拒绝新请求。给正在 decode 的序列留余量，避免"刚放进来就要抢占别人"。
  double watermark = 0.0;
};

class KVCacheManager {
 public:
  explicit KVCacheManager(const KVCacheConfig& cfg);
  KVCacheManager(int num_blocks, int block_size, bool enable_prefix_caching);

  // 新请求进来：为 prompt 分配块（命中前缀缓存的块直接复用）。
  // 块不够（或低于水位线）时返回 false，并且不改变任何状态。
  bool add_sequence(SeqId seq, const std::vector<Token>& prompt);

  // decode 一步：给序列追加一个 token。
  // 需要新块但没有空闲块时返回 false 且不改变状态，调用方应当抢占（preempt）别的序列。
  bool append_token(SeqId seq, Token tok);

  // 并行采样 / beam search：子序列共享父序列的全部物理块（只加引用计数，不拷贝）。
  void fork(SeqId parent, SeqId child);

  // 序列结束或被抢占：归还它占用的块。
  void free_sequence(SeqId seq);

  bool has_sequence(SeqId seq) const { return seqs_.count(seq) > 0; }
  const std::vector<BlockId>& block_table(SeqId seq) const;
  int num_tokens(SeqId seq) const;

  // 取走累计的写时复制操作（执行层每一步调度后取一次）。
  std::vector<CopyOp> take_copy_ops();

  int block_size() const { return block_size_; }
  int num_total_blocks() const { return allocator_.num_total(); }
  // 可用块 = 真正空闲的块 + LRU 里可以被驱逐的缓存块。
  int num_free_blocks() const { return allocator_.num_free() + static_cast<int>(lru_.size()); }
  // 被序列持有（引用计数 > 0）的块。
  int num_used_blocks() const { return num_total_blocks() - num_free_blocks(); }
  // LRU 里引用计数为 0、但内容还留着可以被命中的块。
  int num_evictable_blocks() const { return static_cast<int>(lru_.size()); }
  bool is_evictable(BlockId id) const { return in_lru_.at(id) != 0; }
  int refcount(BlockId id) const { return allocator_.refcount(id); }
  int block_fill(BlockId id) const { return block_fill_.at(id); }

  // 已分配物理块中真正存了 token 的槽位比例，用来衡量内部碎片。
  double slot_utilization() const;
  // 累计有多少个块是通过前缀缓存复用的。
  int64_t prefix_hit_blocks() const { return prefix_hit_blocks_; }
  int64_t cached_block_count() const { return static_cast<int64_t>(cache_.size()); }
  // 累计有多少个缓存块是因为空间不够被 LRU 驱逐的。
  int64_t evicted_blocks() const { return evicted_blocks_; }

 private:
  struct Sequence {
    std::vector<Token> tokens;
    std::vector<BlockId> blocks;   // block table：第 i 个逻辑块 -> 物理块
    std::vector<uint64_t> hashes;  // 每个"写满的"逻辑块对应的前缀哈希
  };

  static uint64_t hash_block(uint64_t prev_hash, const Token* begin, int n);
  // 一个块刚写满时登记到前缀缓存。
  void maybe_register_full_block(Sequence& s, int logical_idx);
  // 归还一个块的引用；如果块被真正释放，同时清掉它的缓存登记和填充计数。
  void release_block(BlockId id);
  std::optional<BlockId> allocate_block();
  // 把块从前缀缓存里摘掉。
  void unregister_block(BlockId id);
  // 命中一个缓存块：在 LRU 里就复活它，否则引用计数加一。
  void acquire_cached_block(BlockId id);

  int block_size_;
  bool enable_prefix_caching_;
  bool lru_eviction_;
  int watermark_blocks_;
  BlockAllocator allocator_;
  std::unordered_map<SeqId, Sequence> seqs_;

  std::unordered_map<uint64_t, BlockId> cache_;    // 前缀哈希 -> 物理块
  std::vector<std::optional<uint64_t>> block_hash_;  // 物理块 -> 它登记的哈希
  std::vector<int> block_fill_;                      // 物理块里已写入的 token 数

  // LRU 驱逐队列：队头是最久没被用过的块。lru_pos_ 用来 O(1) 删除。
  std::list<BlockId> lru_;
  std::vector<std::list<BlockId>::iterator> lru_pos_;
  std::vector<char> in_lru_;

  std::vector<CopyOp> pending_copies_;
  int64_t prefix_hit_blocks_ = 0;
  int64_t evicted_blocks_ = 0;
};

}  // namespace kvc
