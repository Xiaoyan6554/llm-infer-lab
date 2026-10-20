// 显存调度模拟器：同样大小的 KV cache，对比"连续预留"和"分页按需分配"能同时服务多少请求。
//
// 只模拟显存管理，不模拟计算：每一步（一次 decode 迭代）每个运行中的序列生成 1 个 token。
// 所以"平均 batch 大小"和"总步数"直接反映显存管理方式对吞吐的上限影响。
// 计算方面只统计"需要 prefill 的 token 数"和"被抢占后重算的 token 数"，不折算成时间。
//
// 用法：./kv_sim [--requests N] [--capacity-tokens N] [--block-size N]
//               [--max-model-len N] [--max-num-seqs N] [--shared-prefix N] [--seed N]
//               [--tenants N] [--tenant-prefix N] [--tenant-max-seqs N]
//               [--json PATH] [--trace PATH] [--trace-every N]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "kv_cache_manager.h"

using namespace kvc;

namespace {

// 自己写的随机数生成器（splitmix64），保证 Linux / macOS 上跑出来的数字完全一样。
// （std::uniform_int_distribution 在不同标准库里实现不同，结果不可复现。）
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  int uniform(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }  // [0, 1)
};

struct Config {
  int requests = 1000;
  int capacity_tokens = 65536;  // 例：Qwen2.5-7B bf16 每 token KV 约 56KB，65536 token ≈ 3.5GB
  int block_size = 16;
  int max_model_len = 2048;
  int max_num_seqs = 256;
  int shared_prefix = 512;  // 实验二里所有请求共享的 system prompt 长度
  uint64_t seed = 42;
  // 实验五：多个应用各有一段 system prompt，请求按 Zipf 分布落到不同应用上
  int tenants = 64;
  int tenant_prefix = 1024;
  int tenant_max_seqs = 32;
  std::string json_path;
  std::string trace_path;
  int trace_every = 20;
};

struct Request {
  std::vector<Token> prompt;
  int output_len;
};

std::vector<Request> make_workload(const Config& c, int shared_prefix) {
  Rng rng(c.seed);
  std::vector<Token> sys(shared_prefix);
  for (auto& t : sys) t = rng.uniform(0, 150000);
  std::vector<Request> reqs(c.requests);
  for (auto& r : reqs) {
    int max_prompt = c.max_model_len / 2;
    int plen = rng.uniform(32, max_prompt);
    r.output_len = rng.uniform(16, c.max_model_len - max_prompt - shared_prefix);
    r.prompt = sys;
    for (int i = 0; i < plen; ++i) r.prompt.push_back(rng.uniform(0, 150000));
  }
  return reqs;
}

// 多租户：tenants 个不同的 system prompt，第 k 个被选中的概率 ∝ 1/(k+1)（Zipf, s=1）。
// 热门应用的前缀一直有请求在用；冷门应用的请求零星到来，前缀能否留在缓存里就看 LRU。
std::vector<Request> make_tenant_workload(const Config& c) {
  Rng rng(c.seed + 7);
  std::vector<std::vector<Token>> sys(c.tenants, std::vector<Token>(c.tenant_prefix));
  for (auto& s : sys)
    for (auto& t : s) t = rng.uniform(0, 150000);
  std::vector<double> cdf(c.tenants);
  double acc = 0;
  for (int k = 0; k < c.tenants; ++k) cdf[k] = (acc += 1.0 / (k + 1));
  for (auto& x : cdf) x /= acc;
  const int room = c.max_model_len - c.tenant_prefix;
  std::vector<Request> reqs(c.requests);
  for (auto& r : reqs) {
    int k = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), rng.unit()) - cdf.begin());
    k = std::min(k, c.tenants - 1);
    int plen = rng.uniform(32, room / 2);
    r.output_len = rng.uniform(16, room - room / 2);
    r.prompt = sys[k];
    for (int i = 0; i < plen; ++i) r.prompt.push_back(rng.uniform(0, 150000));
  }
  return reqs;
}

struct Result {
  std::string name;
  int block_size = 16;
  double watermark = 0.0;
  int64_t steps = 0;
  int64_t generated = 0;
  double sum_reserved = 0, sum_stored = 0;  // 每步累加，最后求平均
  int64_t preemptions = 0;
  int64_t frag_rejects = 0;  // 总空闲够、但找不到连续空间而拒绝的次数（外部碎片）
  int64_t prefix_hit_blocks = 0;
  int64_t admitted_tokens = 0;   // 所有准入（含被抢占后重新准入）时 prompt 的 token 总数
  int64_t hit_tokens = 0;        // 其中命中前缀缓存、不用重新 prefill 的 token
  int64_t recompute_tokens = 0;  // 被抢占的请求重新准入时，需要重新 prefill 的 token
  int64_t evicted_blocks = 0;
  int64_t sum_table_entries = 0;  // 每步所有运行中序列的块表长度之和，用来算平均块表长度

  double batch() const { return static_cast<double>(generated) / steps; }
  double reserved() const { return sum_reserved / steps; }
  double waste() const {
    double r = reserved();
    return r > 0 ? 1.0 - (sum_stored / steps) / r : 0.0;
  }
  int64_t prefill_tokens() const { return admitted_tokens - hit_tokens; }
  double hit_rate() const { return admitted_tokens ? static_cast<double>(hit_tokens) / admitted_tokens : 0.0; }
  double avg_table_len() const { return generated ? static_cast<double>(sum_table_entries) / generated : 0.0; }
};

// ---------- 动画回放：每隔几步把显存里每个块的状态拍一张快照 ----------
// 每个格子 = block_size 个 token 的显存。状态编码成一个字符：ALPH[state * 10 + color]。
enum CellState { kFree = 0, kReservedEmpty = 1, kPartial = 2, kFull = 3, kShared = 4, kCached = 5 };
const char kAlphabet[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

struct Frame {
  int64_t step;
  std::string grid;
  int running, queued, done;
  double used, stored;
  int64_t preemptions;
};

struct Trace {
  int every = 20;
  int64_t total_steps = 0;
  std::vector<Frame> frames;
};

char cell_char(int state, int owner) {
  int color = (state >= kReservedEmpty && state <= kFull) ? owner % 10 : 0;
  return kAlphabet[state * 10 + color];
}

// ---------- 策略 1/2：连续预留（first-fit 分配连续地址区间） ----------
class ContiguousPool {
 public:
  explicit ContiguousPool(int cap) : cap_(cap) { free_[0] = cap; }
  int64_t alloc(int n) {
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second >= n) {
        int start = it->first, len = it->second;
        free_.erase(it);
        if (len > n) free_[start + n] = len - n;
        free_total_ -= n;
        return start;
      }
    }
    return -1;
  }
  void release(int start, int n) {
    free_total_ += n;
    auto it = free_.emplace(start, n).first;
    auto next = std::next(it);
    if (next != free_.end() && it->first + it->second == next->first) {
      it->second += next->second;
      free_.erase(next);
    }
    if (it != free_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second == it->first) {
        prev->second += it->second;
        free_.erase(it);
      }
    }
  }
  int free_total() const { return free_total_; }

 private:
  int cap_;
  int free_total_ = cap_;
  std::map<int, int> free_;  // 起始地址 -> 长度
};

Result run_contiguous(const Config& c, const std::vector<Request>& reqs, bool oracle, Trace* tr = nullptr) {
  Result res;
  res.name = oracle ? "连续预留(已知输出长度,理想上限)" : "连续预留(按 max_model_len)";
  res.block_size = c.block_size;
  ContiguousPool pool(c.capacity_tokens);
  struct Run { int idx; int64_t start; int reserve; int len; int remaining; };
  std::deque<int> queue;
  for (int i = 0; i < static_cast<int>(reqs.size()); ++i) queue.push_back(i);
  std::vector<Run> running;
  int done = 0;
  while (done < static_cast<int>(reqs.size())) {
    while (!queue.empty() && static_cast<int>(running.size()) < c.max_num_seqs) {
      const Request& r = reqs[queue.front()];
      int plen = static_cast<int>(r.prompt.size());
      int reserve = oracle ? plen + r.output_len : c.max_model_len;
      int64_t start = pool.alloc(reserve);
      if (start < 0) {
        if (pool.free_total() >= reserve) ++res.frag_rejects;
        break;
      }
      running.push_back({queue.front(), start, reserve, plen, r.output_len});
      res.admitted_tokens += plen;
      queue.pop_front();
    }
    double reserved = 0, stored = 0;
    for (auto& s : running) {
      ++s.len;
      --s.remaining;
      reserved += s.reserve;
      stored += s.len;
    }
    res.generated += static_cast<int64_t>(running.size());
    res.sum_reserved += reserved / c.capacity_tokens;
    res.sum_stored += stored / c.capacity_tokens;
    ++res.steps;

    if (tr && (res.steps % tr->every == 0 || res.steps == 1)) {
      const int bs = c.block_size, cells = c.capacity_tokens / bs;
      std::string grid(cells, cell_char(kFree, 0));
      for (auto& s : running) {
        for (int64_t cell = s.start / bs; cell <= (s.start + s.reserve - 1) / bs && cell < cells; ++cell) {
          int64_t lo = std::max<int64_t>(cell * bs, s.start), hi = std::min<int64_t>((cell + 1) * bs, s.start + s.reserve);
          int64_t used_lo = lo, used_hi = std::min<int64_t>(hi, s.start + s.len);
          int64_t stored_in_cell = std::max<int64_t>(0, used_hi - used_lo);
          int state = stored_in_cell == 0 ? kReservedEmpty : (stored_in_cell < hi - lo ? kPartial : kFull);
          grid[cell] = cell_char(state, s.idx);
        }
      }
      tr->frames.push_back({res.steps, grid, static_cast<int>(running.size()), static_cast<int>(queue.size()), done,
                            reserved / c.capacity_tokens, stored / c.capacity_tokens, 0});
    }

    std::vector<Run> still;
    for (auto& s : running) {
      if (s.remaining == 0) {
        pool.release(static_cast<int>(s.start), s.reserve);
        ++done;
      } else {
        still.push_back(s);
      }
    }
    running.swap(still);
  }
  if (tr) tr->total_steps = res.steps;
  return res;
}

// ---------- 策略 3：分页按需分配（KVCacheManager） ----------
struct PagedOpts {
  bool prefix = false;
  bool lru = false;
  double watermark = 0.0;
  int block_size = 16;
  int max_num_seqs = 256;
};

Result run_paged(const Config& c, const std::vector<Request>& reqs, const PagedOpts& o, const std::string& name,
                 Trace* tr = nullptr) {
  Result res;
  res.name = name;
  res.block_size = o.block_size;
  res.watermark = o.watermark;
  const int num_blocks = c.capacity_tokens / o.block_size;
  KVCacheConfig kc;
  kc.num_blocks = num_blocks;
  kc.block_size = o.block_size;
  kc.enable_prefix_caching = o.prefix;
  kc.lru_eviction = o.lru;
  kc.watermark = o.watermark;
  KVCacheManager kv(kc);
  Rng rng(c.seed + 1);

  struct Run { int idx; std::vector<Token> tokens; int remaining; bool preempted; };
  std::deque<Run> queue;
  for (int i = 0; i < static_cast<int>(reqs.size()); ++i) queue.push_back({i, reqs[i].prompt, reqs[i].output_len, false});
  std::vector<Run> running;  // 按进入顺序排列，抢占时踢最后进来的
  int done = 0;

  while (done < static_cast<int>(reqs.size())) {
    while (!queue.empty() && static_cast<int>(running.size()) < o.max_num_seqs) {
      Run& r = queue.front();
      // 被抢占过的请求重新进来时，把已生成的 token 当作 prompt 重新计算（recompute）。
      const int64_t hits_before = kv.prefix_hit_blocks();
      if (!kv.add_sequence(r.idx, r.tokens)) break;
      const int64_t n = static_cast<int64_t>(r.tokens.size());
      const int64_t hit = (kv.prefix_hit_blocks() - hits_before) * o.block_size;
      res.admitted_tokens += n;
      res.hit_tokens += hit;
      if (r.preempted) res.recompute_tokens += n - hit;
      running.push_back(std::move(r));
      queue.pop_front();
    }

    for (size_t i = 0; i < running.size();) {
      Token t = rng.uniform(0, 150000);
      if (kv.append_token(running[i].idx, t)) {
        running[i].tokens.push_back(t);
        --running[i].remaining;
        ++res.generated;
        res.sum_table_entries += static_cast<int64_t>(kv.block_table(running[i].idx).size());
        ++i;
        continue;
      }
      // 没有空闲块：抢占最后进来的序列，释放它的块，放回队首。
      Run victim = std::move(running.back());
      running.pop_back();
      kv.free_sequence(victim.idx);
      victim.preempted = true;
      queue.push_front(std::move(victim));
      ++res.preemptions;
      // 如果被踢的正是当前序列，i 已经越界，循环自然结束；否则重试当前序列。
    }

    const double used = static_cast<double>(kv.num_used_blocks()) * o.block_size / c.capacity_tokens;
    const double stored = kv.slot_utilization() * used;
    res.sum_reserved += used;
    res.sum_stored += stored;
    ++res.steps;

    if (tr && (res.steps % tr->every == 0 || res.steps == 1)) {
      std::string grid(num_blocks, cell_char(kFree, 0));
      for (BlockId b = 0; b < num_blocks; ++b)
        if (kv.is_evictable(b)) grid[b] = cell_char(kCached, 0);
      for (auto& s : running) {
        for (BlockId b : kv.block_table(s.idx)) {
          int state = kv.refcount(b) > 1 ? kShared : (kv.block_fill(b) == o.block_size ? kFull : kPartial);
          grid[b] = cell_char(state, s.idx);
        }
      }
      tr->frames.push_back({res.steps, grid, static_cast<int>(running.size()), static_cast<int>(queue.size()), done,
                            used, stored, res.preemptions});
    }

    std::vector<Run> still;
    for (auto& s : running) {
      if (s.remaining == 0) {
        kv.free_sequence(s.idx);
        ++done;
      } else {
        still.push_back(std::move(s));
      }
    }
    running.swap(still);
  }
  res.prefix_hit_blocks = kv.prefix_hit_blocks();
  res.evicted_blocks = kv.evicted_blocks();
  if (tr) tr->total_steps = res.steps;
  return res;
}

// ---------- 输出：Markdown 表格 ----------
void print_table(const char* title, const std::vector<Result>& rs) {
  std::printf("\n### %s\n\n", title);
  std::printf("| 策略 | 总步数 | 平均 batch | 相对吞吐 | 显存占用率 | 占用中被浪费的比例 | 抢占次数 | 外部碎片拒绝 | 前缀命中块 | 重算 token |\n");
  std::printf("|---|---|---|---|---|---|---|---|---|---|\n");
  const double base = rs[0].batch();
  for (const auto& r : rs) {
    std::printf("| %s | %lld | %.1f | %.2fx | %.1f%% | %.1f%% | %lld | %lld | %lld | %lld |\n", r.name.c_str(),
                static_cast<long long>(r.steps), r.batch(), r.batch() / base, r.reserved() * 100, r.waste() * 100,
                static_cast<long long>(r.preemptions), static_cast<long long>(r.frag_rejects),
                static_cast<long long>(r.prefix_hit_blocks), static_cast<long long>(r.recompute_tokens));
  }
}

void print_watermark_table(const std::vector<Result>& rs) {
  std::printf("\n### 实验三：准入水位线（分页 + 前缀缓存，工作负载同实验二）\n\n");
  std::printf("| 水位线 | 总步数 | 平均 batch | 相对吞吐 | 抢占次数 | 重算 token | 重算占全部 prefill 的比例 |\n");
  std::printf("|---|---|---|---|---|---|---|\n");
  const double base = rs[0].batch();
  for (const auto& r : rs) {
    std::printf("| %.1f%% | %lld | %.1f | %.2fx | %lld | %lld | %.1f%% |\n", r.watermark * 100,
                static_cast<long long>(r.steps), r.batch(), r.batch() / base, static_cast<long long>(r.preemptions),
                static_cast<long long>(r.recompute_tokens),
                r.prefill_tokens() ? 100.0 * r.recompute_tokens / r.prefill_tokens() : 0.0);
  }
}

void print_block_size_table(const std::vector<Result>& rs, double contiguous_batch) {
  std::printf("\n### 实验四：块大小（分页，工作负载同实验一）\n\n");
  std::printf("| 块大小 | 总步数 | 平均 batch | 相对吞吐(对比连续预留) | 占用中被浪费的比例 | 抢占次数 | 平均块表长度 |\n");
  std::printf("|---|---|---|---|---|---|---|\n");
  for (const auto& r : rs) {
    std::printf("| %d | %lld | %.1f | %.2fx | %.1f%% | %lld | %.1f |\n", r.block_size, static_cast<long long>(r.steps),
                r.batch(), r.batch() / contiguous_batch, r.waste() * 100, static_cast<long long>(r.preemptions),
                r.avg_table_len());
  }
}

void print_lru_table(const Config& c, const std::vector<Result>& rs) {
  std::printf("\n### 实验五：LRU 延迟驱逐（%d 个应用各有 %d token 的 system prompt，Zipf 分布，max_num_seqs=%d）\n\n",
              c.tenants, c.tenant_prefix, c.tenant_max_seqs);
  std::printf("| 策略 | 前缀命中率 | 需要 prefill 的 token | 相对无缓存 | 驱逐块数 | 平均 batch | 抢占次数 |\n");
  std::printf("|---|---|---|---|---|---|---|\n");
  const double base = static_cast<double>(rs[0].prefill_tokens());
  for (const auto& r : rs) {
    std::printf("| %s | %.1f%% | %lld | %.2fx | %lld | %.1f | %lld |\n", r.name.c_str(), r.hit_rate() * 100,
                static_cast<long long>(r.prefill_tokens()), r.prefill_tokens() / base,
                static_cast<long long>(r.evicted_blocks), r.batch(), static_cast<long long>(r.preemptions));
  }
}

// ---------- 输出：JSON（给 dashboard 用） ----------
void json_row(FILE* f, const Result& r, double base_batch) {
  std::fprintf(f,
               "{\"name\":\"%s\",\"block_size\":%d,\"watermark\":%.4f,\"steps\":%lld,\"avg_batch\":%.4f,"
               "\"rel_throughput\":%.4f,\"mem_used\":%.4f,\"waste\":%.4f,\"preemptions\":%lld,\"frag_rejects\":%lld,"
               "\"prefix_hit_blocks\":%lld,\"admitted_tokens\":%lld,\"hit_tokens\":%lld,\"prefill_tokens\":%lld,"
               "\"recompute_tokens\":%lld,\"hit_rate\":%.4f,\"evicted_blocks\":%lld,\"avg_table_len\":%.2f}",
               r.name.c_str(), r.block_size, r.watermark, static_cast<long long>(r.steps), r.batch(),
               r.batch() / base_batch, r.reserved(), r.waste(), static_cast<long long>(r.preemptions),
               static_cast<long long>(r.frag_rejects), static_cast<long long>(r.prefix_hit_blocks),
               static_cast<long long>(r.admitted_tokens), static_cast<long long>(r.hit_tokens),
               static_cast<long long>(r.prefill_tokens()), static_cast<long long>(r.recompute_tokens), r.hit_rate(),
               static_cast<long long>(r.evicted_blocks), r.avg_table_len());
}

void json_rows(FILE* f, const char* key, const std::vector<Result>& rs, double base_batch, bool last = false) {
  std::fprintf(f, "\"%s\":[", key);
  for (size_t i = 0; i < rs.size(); ++i) {
    if (i) std::fputc(',', f);
    json_row(f, rs[i], base_batch);
  }
  std::fprintf(f, "]%s\n", last ? "" : ",");
}

void json_config(FILE* f, const Config& c) {
  std::fprintf(f,
               "\"config\":{\"requests\":%d,\"capacity_tokens\":%d,\"block_size\":%d,\"max_model_len\":%d,"
               "\"max_num_seqs\":%d,\"shared_prefix\":%d,\"seed\":%llu,\"tenants\":%d,\"tenant_prefix\":%d,"
               "\"tenant_max_seqs\":%d},\n",
               c.requests, c.capacity_tokens, c.block_size, c.max_model_len, c.max_num_seqs, c.shared_prefix,
               static_cast<unsigned long long>(c.seed), c.tenants, c.tenant_prefix, c.tenant_max_seqs);
}

void json_trace_run(FILE* f, const std::string& name, const Trace& t, bool last) {
  std::fprintf(f, "{\"name\":\"%s\",\"total_steps\":%lld,\"frames\":[\n", name.c_str(),
               static_cast<long long>(t.total_steps));
  for (size_t i = 0; i < t.frames.size(); ++i) {
    const Frame& fr = t.frames[i];
    std::fprintf(f, "%s{\"s\":%lld,\"r\":%d,\"q\":%d,\"d\":%d,\"u\":%.4f,\"st\":%.4f,\"p\":%lld,\"g\":\"%s\"}",
                 i ? ",\n" : "", static_cast<long long>(fr.step), fr.running, fr.queued, fr.done, fr.used, fr.stored,
                 static_cast<long long>(fr.preemptions), fr.grid.c_str());
  }
  std::fprintf(f, "]}%s\n", last ? "" : ",");
}

void write_trace(const Config& c) {
  FILE* f = std::fopen(c.trace_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "无法写入 %s\n", c.trace_path.c_str());
    std::exit(1);
  }
  const int cells = c.capacity_tokens / c.block_size;
  std::fprintf(f, "{\n");
  json_config(f, c);
  std::fprintf(f, "\"cells\":%d,\"every\":%d,\"alphabet\":\"%s\",\n\"scenes\":[\n", cells, c.trace_every, kAlphabet);

  PagedOpts paged;
  paged.block_size = c.block_size;
  paged.max_num_seqs = c.max_num_seqs;
  PagedOpts prefix = paged;
  prefix.prefix = true;

  auto w1 = make_workload(c, 0);
  Trace t1, t2;
  t1.every = t2.every = c.trace_every;
  run_contiguous(c, w1, false, &t1);
  run_paged(c, w1, paged, "分页", &t2);
  std::fprintf(f, "{\"id\":\"paging\",\"runs\":[\n");
  json_trace_run(f, "连续预留(按 max_model_len)", t1, false);
  json_trace_run(f, "分页(PagedAttention 方式)", t2, true);
  std::fprintf(f, "]},\n");

  auto w2 = make_workload(c, c.shared_prefix);
  Trace t3, t4;
  t3.every = t4.every = c.trace_every;
  run_paged(c, w2, paged, "分页", &t3);
  run_paged(c, w2, prefix, "分页 + 前缀缓存", &t4);
  std::fprintf(f, "{\"id\":\"prefix\",\"runs\":[\n");
  json_trace_run(f, "分页(无前缀缓存)", t3, false);
  json_trace_run(f, "分页 + 前缀缓存", t4, true);
  std::fprintf(f, "]}\n]}\n");
  std::fclose(f);
  std::printf("动画数据已写入 %s（%d 个格子，每 %d 步一帧）\n", c.trace_path.c_str(), cells, c.trace_every);
}

Config parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::printf(
          "用法: kv_sim [--requests 1000] [--capacity-tokens 65536] [--block-size 16]\n"
          "             [--max-model-len 2048] [--max-num-seqs 256] [--shared-prefix 512] [--seed 42]\n"
          "             [--tenants 64] [--tenant-prefix 1024] [--tenant-max-seqs 32]\n"
          "             [--json results/sim.json]   把全部实验结果写成 JSON\n"
          "             [--trace results/trace.json] [--trace-every 20]  只导出显存动画数据\n");
      std::exit(0);
    }
  }
  if (argc % 2 == 0) {
    std::fprintf(stderr, "参数需要成对出现：--名字 数值\n");
    std::exit(2);
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    const char* s = argv[i + 1];
    long v = std::strtol(s, nullptr, 10);
    if (k == "--requests") c.requests = static_cast<int>(v);
    else if (k == "--capacity-tokens") c.capacity_tokens = static_cast<int>(v);
    else if (k == "--block-size") c.block_size = static_cast<int>(v);
    else if (k == "--max-model-len") c.max_model_len = static_cast<int>(v);
    else if (k == "--max-num-seqs") c.max_num_seqs = static_cast<int>(v);
    else if (k == "--shared-prefix") c.shared_prefix = static_cast<int>(v);
    else if (k == "--seed") c.seed = static_cast<uint64_t>(v);
    else if (k == "--tenants") c.tenants = static_cast<int>(v);
    else if (k == "--tenant-prefix") c.tenant_prefix = static_cast<int>(v);
    else if (k == "--tenant-max-seqs") c.tenant_max_seqs = static_cast<int>(v);
    else if (k == "--json") c.json_path = s;
    else if (k == "--trace") c.trace_path = s;
    else if (k == "--trace-every") c.trace_every = static_cast<int>(v);
    else {
      std::fprintf(stderr, "unknown flag %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (c.capacity_tokens % c.block_size != 0 || c.trace_every <= 0) {
    std::fprintf(stderr, "capacity-tokens 必须是 block-size 的整数倍，trace-every 必须 > 0\n");
    std::exit(2);
  }
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  Config c = parse(argc, argv);
  std::printf("配置: requests=%d capacity_tokens=%d block_size=%d max_model_len=%d max_num_seqs=%d seed=%llu\n",
              c.requests, c.capacity_tokens, c.block_size, c.max_model_len, c.max_num_seqs,
              static_cast<unsigned long long>(c.seed));
  if (!c.trace_path.empty()) {
    write_trace(c);
    return 0;
  }

  PagedOpts paged;
  paged.block_size = c.block_size;
  paged.max_num_seqs = c.max_num_seqs;
  PagedOpts prefix = paged;
  prefix.prefix = true;
  PagedOpts prefix_lru = prefix;
  prefix_lru.lru = true;

  // 实验一
  auto w1 = make_workload(c, 0);
  PagedOpts paged_wm = paged;
  paged_wm.watermark = 0.01;
  std::vector<Result> exp1 = {run_contiguous(c, w1, false), run_contiguous(c, w1, true),
                              run_paged(c, w1, paged, "分页(PagedAttention 方式)"),
                              run_paged(c, w1, paged_wm, "分页 + 1% 准入水位线")};
  print_table("实验一：连续预留 vs 分页分配（请求之间没有公共前缀）", exp1);

  // 实验二
  auto w2 = make_workload(c, c.shared_prefix);
  std::vector<Result> exp2 = {run_paged(c, w2, paged, "分页(PagedAttention 方式)"),
                              run_paged(c, w2, prefix, "分页 + 前缀缓存")};
  char title[200];
  std::snprintf(title, sizeof(title), "实验二：前缀缓存（所有请求共享 %d token 的 system prompt）", c.shared_prefix);
  print_table(title, exp2);

  // 实验三：准入水位线
  std::vector<Result> exp3;
  for (double wm : {0.0, 0.005, 0.01, 0.02, 0.05, 0.10}) {
    PagedOpts o = prefix;
    o.watermark = wm;
    exp3.push_back(run_paged(c, w2, o, "分页 + 前缀缓存"));
  }
  print_watermark_table(exp3);

  // 实验四：块大小
  std::vector<Result> exp4;
  for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    if (c.capacity_tokens % bs) continue;
    PagedOpts o = paged;
    o.block_size = bs;
    exp4.push_back(run_paged(c, w1, o, "分页"));
  }
  print_block_size_table(exp4, exp1[0].batch());

  // 实验五：LRU 延迟驱逐
  auto w5 = make_tenant_workload(c);
  PagedOpts t_plain = paged, t_prefix = prefix, t_lru = prefix_lru;
  t_plain.max_num_seqs = t_prefix.max_num_seqs = t_lru.max_num_seqs = c.tenant_max_seqs;
  std::vector<Result> exp5 = {run_paged(c, w5, t_plain, "分页(无前缀缓存)"),
                              run_paged(c, w5, t_prefix, "前缀缓存(引用归零立即失效)"),
                              run_paged(c, w5, t_lru, "前缀缓存 + LRU 延迟驱逐")};
  print_lru_table(c, exp5);

  if (!c.json_path.empty()) {
    FILE* f = std::fopen(c.json_path.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "无法写入 %s\n", c.json_path.c_str());
      return 1;
    }
    std::fprintf(f, "{\n");
    json_config(f, c);
    json_rows(f, "exp1_paging", exp1, exp1[0].batch());
    json_rows(f, "exp2_prefix", exp2, exp2[0].batch());
    json_rows(f, "exp3_watermark", exp3, exp3[0].batch());
    json_rows(f, "exp4_block_size", exp4, exp1[0].batch());
    json_rows(f, "exp5_lru", exp5, exp5[0].batch(), true);
    std::fprintf(f, "}\n");
    std::fclose(f);
    std::printf("\nJSON 已写入 %s\n", c.json_path.c_str());
  }
  return 0;
}
