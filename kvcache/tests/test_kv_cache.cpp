// 单元测试：不依赖 gtest，自带一个极简测试框架，clone 下来就能编译运行。
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "kv_cache_manager.h"

using namespace kvc;

namespace {
struct TestCase {
  std::string name;
  std::function<void()> fn;
};
std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}
int g_failures = 0;
struct Registrar {
  Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
}  // namespace

#define TEST(name)                                   \
  static void name();                                \
  static Registrar reg_##name(#name, name);          \
  static void name()

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
      ++g_failures;                                                          \
    }                                                                        \
  } while (0)

static std::vector<Token> seq_tokens(int n, int start = 0) {
  std::vector<Token> t(n);
  for (int i = 0; i < n; ++i) t[i] = start + i;
  return t;
}

TEST(allocator_basic) {
  BlockAllocator a(3);
  CHECK(a.num_free() == 3);
  auto b0 = a.allocate(), b1 = a.allocate(), b2 = a.allocate();
  CHECK(b0 && b1 && b2);
  CHECK(!a.allocate());  // 用完了
  a.incref(*b1);
  CHECK(a.refcount(*b1) == 2);
  CHECK(!a.decref(*b1));  // 还有一个引用
  CHECK(a.decref(*b1));   // 归零
  CHECK(a.num_free() == 1);
  CHECK(a.allocate() == b1);  // 刚释放的块被复用
}

TEST(add_sequence_allocates_ceil_blocks) {
  KVCacheManager m(10, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(9)));  // 9 个 token，块大小 4 -> 3 块
  CHECK(m.block_table(1).size() == 3);
  CHECK(m.num_free_blocks() == 7);
  // 9 个 token / 12 个槽位
  CHECK(std::fabs(m.slot_utilization() - 9.0 / 12.0) < 1e-9);
}

TEST(add_sequence_fails_without_side_effects) {
  KVCacheManager m(2, 4, false);
  CHECK(!m.add_sequence(1, seq_tokens(9)));  // 需要 3 块，只有 2 块
  CHECK(!m.has_sequence(1));
  CHECK(m.num_free_blocks() == 2);
}

TEST(append_crosses_block_boundary) {
  KVCacheManager m(10, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(4)));  // 正好 1 块写满
  CHECK(m.block_table(1).size() == 1);
  CHECK(m.append_token(1, 100));  // 第 5 个 token 需要新块
  CHECK(m.block_table(1).size() == 2);
  for (int i = 0; i < 3; ++i) CHECK(m.append_token(1, 101 + i));
  CHECK(m.block_table(1).size() == 2);  // 第 2 块写满，还不需要第 3 块
  CHECK(m.num_tokens(1) == 8);
}

TEST(append_fails_when_out_of_blocks) {
  KVCacheManager m(1, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(4)));
  CHECK(!m.append_token(1, 7));  // 没块了
  CHECK(m.num_tokens(1) == 4);   // 状态没有变
  CHECK(m.block_table(1).size() == 1);
}

TEST(fork_shares_blocks_and_copy_on_write) {
  KVCacheManager m(10, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(6)));  // 块: [满][2/4]
  m.fork(1, 2);
  CHECK(m.num_used_blocks() == 2);  // fork 不分配新块
  BlockId shared_last = m.block_table(1).back();
  CHECK(m.refcount(shared_last) == 2);

  CHECK(m.append_token(1, 50));  // 父序列写共享块 -> 触发 COW
  auto ops = m.take_copy_ops();
  CHECK(ops.size() == 1);
  CHECK(ops[0].src == shared_last);
  CHECK(m.block_table(1).back() == ops[0].dst);
  CHECK(m.refcount(shared_last) == 1);
  CHECK(m.num_used_blocks() == 3);

  CHECK(m.append_token(2, 60));  // 子序列现在独占这个块，原地写
  CHECK(m.take_copy_ops().empty());
  CHECK(m.block_table(2).back() == shared_last);
  CHECK(m.block_table(1).front() == m.block_table(2).front());  // 第一个满块一直共享
}

TEST(prefix_caching_reuses_full_blocks) {
  KVCacheManager m(20, 4, true);
  std::vector<Token> sys = seq_tokens(8);  // 2 个满块的公共前缀（类似 system prompt）
  std::vector<Token> a = sys, b = sys;
  a.insert(a.end(), {900, 901});
  b.insert(b.end(), {800, 801, 802});

  CHECK(m.add_sequence(1, a));
  CHECK(m.num_used_blocks() == 3);
  CHECK(m.add_sequence(2, b));
  CHECK(m.prefix_hit_blocks() == 2);
  CHECK(m.num_used_blocks() == 4);  // 只为 b 的尾巴新分配 1 块
  CHECK(m.block_table(1)[0] == m.block_table(2)[0]);
  CHECK(m.block_table(1)[1] == m.block_table(2)[1]);
  CHECK(m.block_table(1)[2] != m.block_table(2)[2]);
}

TEST(prefix_caching_requires_identical_prefix) {
  KVCacheManager m(20, 4, true);
  std::vector<Token> a = seq_tokens(8);
  std::vector<Token> b = seq_tokens(8);
  b[0] = 999;  // 第一个块不同 -> 第二个块即使内容相同也不能复用（哈希链）
  CHECK(m.add_sequence(1, a));
  CHECK(m.add_sequence(2, b));
  CHECK(m.prefix_hit_blocks() == 0);
  CHECK(m.num_used_blocks() == 4);
}

TEST(prefix_caching_disabled_never_shares) {
  KVCacheManager m(20, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(8)));
  CHECK(m.add_sequence(2, seq_tokens(8)));
  CHECK(m.prefix_hit_blocks() == 0);
  CHECK(m.num_used_blocks() == 4);
}

TEST(decode_filled_block_becomes_cacheable) {
  KVCacheManager m(20, 4, true);
  CHECK(m.add_sequence(1, seq_tokens(2)));
  CHECK(m.append_token(1, 2));
  CHECK(m.append_token(1, 3));  // 块在 decode 阶段被写满，登记进缓存
  CHECK(m.cached_block_count() == 1);
  CHECK(m.add_sequence(2, seq_tokens(5)));
  CHECK(m.prefix_hit_blocks() == 1);
}

TEST(free_returns_all_blocks_and_clears_cache) {
  KVCacheManager m(20, 4, true);
  CHECK(m.add_sequence(1, seq_tokens(10)));
  CHECK(m.add_sequence(2, seq_tokens(10)));
  m.fork(2, 3);
  m.free_sequence(1);
  m.free_sequence(2);
  CHECK(m.num_used_blocks() == 3);  // 序列 3 还持有 3 个块
  m.free_sequence(3);
  CHECK(m.num_free_blocks() == 20);
  CHECK(m.cached_block_count() == 0);
  CHECK(m.slot_utilization() == 0.0);
}

TEST(slot_utilization_after_fork_and_cow) {
  KVCacheManager m(10, 4, false);
  CHECK(m.add_sequence(1, seq_tokens(6)));  // 块: [4/4][2/4]
  m.fork(1, 2);
  // 共享的物理块只算一次：还是 6 个 token / 8 个槽位
  CHECK(std::fabs(m.slot_utilization() - 6.0 / 8.0) < 1e-9);
  CHECK(m.append_token(1, 50));  // COW：父序列拿到一份 3/4 的拷贝，子序列独占原来 2/4 的块
  CHECK(m.num_used_blocks() == 3);
  CHECK(std::fabs(m.slot_utilization() - 9.0 / 12.0) < 1e-9);
}

static KVCacheConfig lru_config(int num_blocks, int block_size) {
  KVCacheConfig c;
  c.num_blocks = num_blocks;
  c.block_size = block_size;
  c.enable_prefix_caching = true;
  c.lru_eviction = true;
  return c;
}

TEST(lru_keeps_freed_blocks_hittable) {
  KVCacheManager m(lru_config(10, 4));
  std::vector<Token> a = seq_tokens(8);
  a.push_back(900);
  CHECK(m.add_sequence(1, a));
  m.free_sequence(1);  // 请求 A 结束
  CHECK(m.num_free_blocks() == 10);      // 对新请求来说，所有块都可用
  CHECK(m.num_evictable_blocks() == 2);  // 两个满块留在 LRU 里，内容还在
  CHECK(m.cached_block_count() == 2);

  std::vector<Token> b = seq_tokens(8);
  b.push_back(800);
  CHECK(m.add_sequence(2, b));  // 内容相同的请求 B 依然命中缓存
  CHECK(m.prefix_hit_blocks() == 2);
  CHECK(m.num_evictable_blocks() == 0);
  CHECK(m.num_used_blocks() == 3);

  // 对照：不开 LRU 时，A 一结束缓存就失效了
  KVCacheManager plain(10, 4, true);
  CHECK(plain.add_sequence(1, a));
  plain.free_sequence(1);
  CHECK(plain.add_sequence(2, b));
  CHECK(plain.prefix_hit_blocks() == 0);
}

TEST(lru_evicts_oldest_only_when_no_free_block) {
  KVCacheManager m(lru_config(4, 4));
  CHECK(m.add_sequence(1, seq_tokens(4, 0)));    // 块 A
  m.free_sequence(1);                            // LRU: [A]
  CHECK(m.add_sequence(2, seq_tokens(4, 100)));  // 块 B：优先用真正空闲的块，不驱逐 A
  CHECK(m.evicted_blocks() == 0);
  m.free_sequence(2);                            // LRU: [A, B]
  CHECK(m.add_sequence(3, seq_tokens(12, 200)));  // 要 3 块：2 个空闲 + 驱逐最老的 A
  CHECK(m.evicted_blocks() == 1);
  CHECK(m.add_sequence(4, seq_tokens(4, 100)));  // B 还在，命中
  CHECK(m.prefix_hit_blocks() == 1);
  m.free_sequence(3);
  m.free_sequence(4);
  CHECK(m.add_sequence(5, seq_tokens(4, 0)));  // A 已经被驱逐，不会命中
  CHECK(m.prefix_hit_blocks() == 1);
}

TEST(lru_hits_are_not_counted_as_free_space) {
  KVCacheManager m(lru_config(3, 4));
  CHECK(m.add_sequence(1, seq_tokens(8)));
  m.free_sequence(1);  // 2 块在 LRU，1 块空闲
  // 命中 LRU 里的 2 块，还要 2 个新块，但复活之后只剩 1 块可用 -> 失败且不改状态
  std::vector<Token> big = seq_tokens(8);
  big.insert(big.end(), {1, 2, 3, 4, 5});
  CHECK(!m.add_sequence(2, big));
  CHECK(!m.has_sequence(2));
  CHECK(m.num_evictable_blocks() == 2);
  CHECK(m.prefix_hit_blocks() == 0);
  std::vector<Token> fits = seq_tokens(8);
  fits.insert(fits.end(), {1, 2, 3});
  CHECK(m.add_sequence(2, fits));
  CHECK(m.prefix_hit_blocks() == 2);
  CHECK(m.num_free_blocks() == 0);
}

TEST(lru_free_all_keeps_cache_but_releases_space) {
  KVCacheManager m(lru_config(20, 4));
  CHECK(m.add_sequence(1, seq_tokens(10)));
  m.fork(1, 2);
  CHECK(m.append_token(2, 77));  // 子序列写共享的尾块 -> COW
  m.free_sequence(1);
  m.free_sequence(2);
  CHECK(m.num_used_blocks() == 0);
  CHECK(m.num_free_blocks() == 20);
  CHECK(m.cached_block_count() == 2);
  CHECK(m.slot_utilization() == 0.0);
}

TEST(watermark_rejects_admission_but_not_decode) {
  KVCacheConfig c;
  c.num_blocks = 100;
  c.block_size = 4;
  c.watermark = 0.1;  // 留 10 块
  KVCacheManager m(c);
  CHECK(m.add_sequence(1, seq_tokens(80 * 4)));  // 80 块，剩 20
  CHECK(!m.add_sequence(2, seq_tokens(11 * 4)));  // 放进去只剩 9 块 < 10，拒绝
  CHECK(!m.has_sequence(2));
  CHECK(m.add_sequence(3, seq_tokens(10 * 4)));  // 剩 10 块，刚好可以
  for (int i = 0; i < 10 * 4; ++i) CHECK(m.append_token(1, i));  // decode 可以用掉水位线以下的块
  CHECK(m.num_free_blocks() == 0);
}

TEST(watermark_ignored_when_nothing_is_running) {
  KVCacheConfig c;
  c.num_blocks = 10;
  c.block_size = 4;
  c.watermark = 0.5;
  KVCacheManager m(c);
  CHECK(m.add_sequence(1, seq_tokens(8 * 4)));  // 系统是空的，不需要给谁留余量
  CHECK(m.num_free_blocks() == 2);
}

int main() {
  int failed_tests = 0;
  for (auto& t : registry()) {
    int before = g_failures;
    t.fn();
    bool ok = g_failures == before;
    if (!ok) ++failed_tests;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", t.name.c_str());
  }
  std::printf("\n%zu tests, %d failed\n", registry().size(), failed_tests);
  return failed_tests == 0 ? 0 : 1;
}
