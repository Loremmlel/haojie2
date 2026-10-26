#include "model.h"
#include <ATen/record_function.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <cmath>
#include <map>
#include <tuple>
#if defined(__AVX2__) || defined(_M_X64)
#include <immintrin.h>
#define HJ_AVX2
#endif

namespace hj {
namespace {
using BF = c10::BFloat16;
using Vector = std::vector<BF>;
#ifdef HJ_AVX2
__m256 load(const BF *p) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p))), 16));
}
void store(BF *p, __m256 x) {
  auto n = _mm256_castps_si256(x);
  auto rounded = _mm256_srli_epi32(
      _mm256_add_epi32(
          n, _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                              _mm256_and_si256(_mm256_srli_epi32(n, 16), _mm256_set1_epi32(1)))),
      16);
  _mm_storeu_si128(
      reinterpret_cast<__m128i *>(p),
      _mm_packus_epi32(_mm256_castsi256_si128(rounded), _mm256_extracti128_si256(rounded, 1)));
}
float horizontal(__m256 acc) {
  auto part = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  part = _mm_hadd_ps(part, part);
  part = _mm_hadd_ps(part, part);
  return _mm_cvtss_f32(part);
}
#endif
float dot(const BF *a, const BF *b, int64_t count) {
  int64_t i = 0;
  float sum = 0;
#ifdef HJ_AVX2
  auto acc = _mm256_setzero_ps();
  for (; i + 8 <= count; i += 8)
    acc = _mm256_fmadd_ps(load(a + i), load(b + i), acc);
  sum = horizontal(acc);
#endif
  for (; i < count; ++i)
    sum += float(a[i]) * float(b[i]);
  return sum;
}
void linear(const BF *input, const T &weight, const T &bias, int64_t offset, int64_t columns,
            BF *out, int64_t rows) {
  auto w = weight.data_ptr<BF>();
  auto b = bias.defined() ? bias.data_ptr<BF>() : nullptr;
  for (int64_t j = 0; j < rows; ++j)
    out[j] = BF(dot(input, w + j * weight.size(1) + offset, columns) + (b ? float(b[j]) : 0));
}
// 四个候选复用两行权重；八路累加和最终归约顺序与单行 dot 完全相同。
void linear_rows(const Vector &input, const T &weight, const T &bias, Vector &out, int64_t count) {
  auto columns = weight.size(1), rows = weight.size(0);
  int64_t first = 0;
#ifdef HJ_AVX2
  auto w = weight.data_ptr<BF>(), b = bias.data_ptr<BF>();
  for (; first + 4 <= count; first += 4) {
    int64_t j = 0;
    for (; j + 2 <= rows; j += 2) {
      __m256 acc[4][2];
      for (auto &pair : acc)
        pair[0] = pair[1] = _mm256_setzero_ps();
      int64_t k = 0;
      for (; k + 8 <= columns; k += 8) {
        auto a = load(w + j * columns + k), z = load(w + (j + 1) * columns + k);
        for (int r = 0; r < 4; ++r) {
          auto x = load(input.data() + (first + r) * columns + k);
          acc[r][0] = _mm256_fmadd_ps(x, a, acc[r][0]);
          acc[r][1] = _mm256_fmadd_ps(x, z, acc[r][1]);
        }
      }
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 2; ++c) {
          float sum = horizontal(acc[r][c]);
          for (auto tail = k; tail < columns; ++tail)
            sum += float(input[(first + r) * columns + tail]) * float(w[(j + c) * columns + tail]);
          out[(first + r) * rows + j + c] = BF(sum + float(b[j + c]));
        }
    }
    for (; j < rows; ++j)
      for (int r = 0; r < 4; ++r)
        out[(first + r) * rows + j] =
            BF(dot(input.data() + (first + r) * columns, w + j * columns, columns) + float(b[j]));
  }
#endif
  for (; first < count; ++first)
    linear(input.data() + first * columns, weight, bias, 0, columns, out.data() + first * rows,
           rows);
}
BF gelu(BF value) {
  static const auto table = [] {
    std::array<uint16_t, 65536> values;
    for (size_t i = 0; i < values.size(); ++i) {
      BF input;
      input.x = uint16_t(i);
      auto x = float(input);
      values[i] = BF(0.5f * x * (1.0f + std::erf(x * 0.7071067811865475244f))).x;
    }
    return values;
  }();
  BF out;
  out.x = table[value.x];
  return out;
}
Vector project(const std::vector<int64_t> &bits, const T &weight, int64_t width) {
  Vector out(width);
  auto w = weight.data_ptr<BF>();
  int64_t col = 0;
#ifdef HJ_AVX2
  for (; col + 8 <= width; col += 8) {
    auto acc = _mm256_setzero_ps();
    for (auto bit : bits)
      acc = _mm256_add_ps(acc, load(w + bit * width + col));
    store(out.data() + col, acc);
  }
#endif
  for (; col < width; ++col) {
    float sum = 0;
    for (auto bit : bits)
      sum += float(w[bit * width + col]);
    out[col] = BF(sum);
  }
  return out;
}
std::vector<uint16_t> normalize(const Vector &logits, const std::vector<uint8_t> &mask) {
  Vector masses(logits.size());
  float maximum = -INFINITY;
  for (size_t i = 0; i < logits.size(); ++i)
    if (mask[i])
      maximum = std::max(maximum, float(logits[i]));
  float sum = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    masses[i] = mask[i] ? BF(std::exp(float(BF(float(logits[i]) - maximum)))) : BF(0);
    sum += float(masses[i]);
  }
  BF denominator(sum);
  TORCH_CHECK(std::isfinite(sum) && float(denominator) > 0, "invalid CPU categorical mass");
  std::vector<uint16_t> out;
  for (auto mass : masses)
    out.push_back(BF(float(mass) / float(denominator)).x);
  return out;
}
template <class V> struct ProjectionCache {
  struct Entry {
    InputRef<V> source;
    Vector value;
  };
  std::unordered_map<const InputValue<V> *, Entry> entries;
  StorageBytes storage;
  void clear() {
    entries.clear();
    storage.set(entries.bucket_count() * 8);
  }
  void prepare(size_t incoming) {
    if (entries.size() + incoming > 4096)
      clear();
  }
  template <class F> const Vector &get(const InputRef<V> &source, int64_t width, F compute) {
    auto found = entries.find(source.get());
    if (found != entries.end())
      return found->second.value;
    auto [entry, inserted] = entries.emplace(source.get(), Entry{source, compute(source->value)});
    storage.set(entries.size() * (sizeof(Entry) + 40 + width * 2) + entries.bucket_count() * 8);
    return entry->second.value;
  }
};
} // namespace

struct CpuPolicy::Impl {
  struct Key {
    std::shared_ptr<Root> root;
    ActionRef action;
    Vector values;
  };
  CpuSnapshot current;
  ProjectionCache<Row> rows;
  ProjectionCache<Action> actions;
  std::map<std::tuple<const Root *, const InputValue<Action> *, int32_t, int32_t>, Key> keys;
  StorageBytes key_storage;
};
CpuPolicy::CpuPolicy() : impl(std::make_unique<Impl>()) {}
CpuPolicy::~CpuPolicy() = default;
void CpuPolicy::clear() {
  impl = std::make_unique<Impl>();
}
CpuSnapshot CpuPolicy::snapshot(Network &model, uint64_t version) {
  torch::NoGradGuard guard;
  auto out = std::make_shared<CpuWeights>();
  out->version = version;
  out->revision = model->revision;
  out->owner = model.ptr().get();
  out->width = model->width;
  auto &query = model->query->at<torch::nn::LinearImpl>(0);
  auto &query_out = model->query->at<torch::nn::LinearImpl>(2);
  auto &key = model->key->at<torch::nn::LinearImpl>(0);
  auto &key_out = model->key->at<torch::nn::LinearImpl>(2);
  auto &value = model->value->at<torch::nn::LinearImpl>(0);
  auto &value_out = model->value->at<torch::nn::LinearImpl>(2);
  std::array<T, 21> sources{model->entity->bits->weight,
                            model->entity->continuous->weight,
                            model->entity->continuous->bias,
                            model->action->bits->weight,
                            model->action->bias,
                            model->context->weight,
                            model->context->bias,
                            query.weight,
                            query.bias,
                            query_out.weight,
                            query_out.bias,
                            key.weight,
                            key.bias,
                            key_out.weight,
                            key_out.bias,
                            model->prior->weight,
                            model->prior->bias,
                            value.weight,
                            value.bias,
                            value_out.weight,
                            value_out.bias};
  std::array<T *, 21> destinations{&out->rows,         &out->continuous,     &out->row_bias,
                                   &out->actions,      &out->action_bias,    &out->context,
                                   &out->context_bias, &out->query,          &out->query_bias,
                                   &out->query_out,    &out->query_out_bias, &out->key,
                                   &out->key_bias,     &out->key_out,        &out->key_out_bias,
                                   &out->prior,        &out->prior_bias,     &out->value,
                                   &out->value_bias,   &out->value_out,      &out->value_out_bias};
  std::vector<T> pieces;
  for (auto &source : sources)
    pieces.push_back(source.reshape({-1}));
  auto flat = torch::cat(pieces);
  auto device = flat.device();
  auto host = torch::empty(flat.sizes(),
                           flat.options().device(torch::kCPU).pinned_memory(device.is_cuda()));
  host.copy_(flat, true);
  // 整份不可变快照只等待一次传输，所有头部张量在 CPU 上共享这块 BF16 存储。
  if (device.is_cuda()) {
    c10::impl::VirtualGuardImpl implementation(device.type());
    implementation.synchronizeStream(implementation.getStream(device));
  }
  int64_t offset = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    *destinations[i] = host.narrow(0, offset, sources[i].numel()).view(sources[i].sizes());
    offset += sources[i].numel();
  }
  return out;
}
std::vector<uint16_t> CpuPolicy::score(const Node &node, const CpuSnapshot &weights,
                                       double *value) {
  RECORD_FUNCTION("haojie::cpu_policy", std::vector<c10::IValue>());
  auto &s = *impl;
  if (s.current != weights) {
    s.rows.clear();
    s.actions.clear();
    s.keys.clear();
    s.key_storage.set(0);
    s.current = weights;
  }
  auto &w = *weights;
  auto width = w.width;
  // 在整次请求之前腾出缓存，保留当前上下文归约使用的所有行引用。
  s.rows.prepare(node.context.size());
  s.actions.prepare(node.candidates.size());
  if (s.keys.size() + node.candidates.size() > 4096)
    s.keys.clear();
  T tokens, context;
  {
    std::lock_guard lock(node.root->cache_mutex);
    TORCH_CHECK(node.root->cached_owner == w.owner && node.root->cached_revision == w.revision &&
                    node.root->host_tokens.defined() && node.root->host_context.defined(),
                "CPU policy root revision differs");
    tokens = node.root->host_tokens;
    context = node.root->host_context;
  }
  TORCH_CHECK(!node.context.empty() && node.candidates.size() == node.mask.size(),
              "invalid CPU policy table");
  std::vector<const Vector *> rows;
  for (auto &row : node.context)
    rows.push_back(&s.rows.get(row, width, [&](const Row &r) {
      TORCH_CHECK((r.present >> 52) == 0 && (r.integers >> 52) == 0, "invalid CPU row flags");
      std::vector<int64_t> bits;
      for (size_t i = 0; i < r.ids.size(); ++i)
        packing::set_bits(bits, packing::integer_bits(r.ids[i]), i * 64);
      packing::set_bits(bits, r.present, 60 * 64);
      packing::set_bits(bits, r.integers, 60 * 64 + 52);
      auto out = project(bits, w.rows, width);
      bool continuous =
          std::any_of(r.values.begin(), r.values.end(), [](auto v) { return (v & 0x7fff) != 0; });
      Vector extra(width);
      if (continuous)
        linear(reinterpret_cast<const BF *>(r.values.data()), w.continuous, w.row_bias, 0, 52,
               extra.data(), width);
      auto bias = w.row_bias.data_ptr<BF>();
      for (int64_t j = 0; j < width; ++j)
        out[j] = gelu(BF(float(out[j]) + float(continuous ? extra[j] : bias[j])));
      return out;
    }));
  Vector mean(width), projected(width), ctx(width), query_hidden(width), query(width);
  for (int64_t j = 0; j < width; ++j) {
    float sum = 0;
    for (auto row : rows)
      sum += float((*row)[j]);
    mean[j] = BF(sum / float(rows.size()));
  }
  linear(mean.data(), w.context, w.context_bias, width, width, projected.data(), width);
  auto root = context.data_ptr<BF>();
  for (int64_t j = 0; j < width; ++j)
    ctx[j] = gelu(BF(float(root[j]) + float(projected[j])));
  if (value) {
    Vector hidden(width);
    linear(ctx.data(), w.value, w.value_bias, 0, width, hidden.data(), width);
    for (auto &x : hidden)
      x = gelu(x);
    BF output;
    linear(hidden.data(), w.value_out, w.value_out_bias, 0, width, &output, 1);
    *value = float(BF(std::tanh(float(output))));
  }
  // 假想分支的空根只取价值估计作搜索截止，不补造成终局或向实局返回空分布。
  if (std::none_of(node.mask.begin(), node.mask.end(), [](auto v) { return v != 0; }))
    return {};
  linear(ctx.data(), w.query, w.query_bias, 0, width, query_hidden.data(), width);
  for (auto &x : query_hidden)
    x = gelu(x);
  linear(query_hidden.data(), w.query_out, w.query_out_bias, 0, width, query.data(), width);
  Vector hidden, logits(node.mask.size());
  std::vector<Impl::Key *> missing;
  std::vector<const Vector *> keys(node.mask.size());
  auto entities = tokens.data_ptr<BF>();
  auto rows_count = tokens.size(0);
  for (size_t i = 0; i < node.candidates.size(); ++i) {
    if (!node.mask[i])
      continue;
    auto source = node.sources[i], target = node.targets[i];
    TORCH_CHECK(source >= -1 && target >= -1 && source < rows_count && target < rows_count,
                "CPU policy entity reference differs");
    auto [entry, inserted] = s.keys.try_emplace(
        std::make_tuple(node.root.get(), node.candidates[i].get(), source, target),
        Impl::Key{node.root, node.candidates[i], {}});
    keys[i] = &entry->second.values;
    if (!inserted)
      continue;
    auto &candidate = s.actions.get(node.candidates[i], width, [&](const Action &action) {
      std::vector<int64_t> bits;
      for (size_t j = 0; j < action.size(); ++j)
        packing::set_bits(bits, packing::integer_bits(action[j]), j * 64);
      auto embedding = project(bits, w.actions, width);
      auto bias = w.action_bias.data_ptr<BF>();
      for (int64_t j = 0; j < width; ++j)
        embedding[j] = gelu(BF(float(embedding[j]) + float(bias[j])));
      Vector out(width);
      linear(embedding.data(), w.key, w.key_bias, 0, width, out.data(), width);
      return out;
    });
    for (int64_t j = 0; j < width; ++j) {
      auto x = BF(float(candidate[j]) +
                  (source < 0 ? 0 : float(entities[int64_t(source) * width * 2 + j])));
      x = BF(float(x) +
             (target < 0 ? 0 : float(entities[int64_t(target) * width * 2 + width + j])));
      hidden.push_back(gelu(x));
    }
    missing.push_back(&entry->second);
  }
  Vector next(missing.size() * width);
  linear_rows(hidden, w.key_out, w.key_out_bias, next, missing.size());
  for (size_t i = 0; i < missing.size(); ++i) {
    auto &out = missing[i]->values;
    out.assign(next.begin() + i * width, next.begin() + (i + 1) * width);
    out.push_back(BF(0));
    linear(out.data(), w.prior, w.prior_bias, 0, width, &out[width], 1);
  }
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!node.mask[i])
      continue;
    auto &key = *keys[i];
    float sum = 0;
    for (int64_t j = 0; j < width; ++j)
      sum += float(BF(float(query[j]) * float(key[j])));
    auto scaled = BF(float(BF(sum)) / float(std::sqrt(double(width))));
    logits[i] = BF(float(scaled) + float(key[width]));
  }
  s.key_storage.set(s.keys.size() * (sizeof(Impl::Key) + 80 + (width + 1) * 2));
  return normalize(logits, node.mask);
}
void cpu_policy_checks(const torch::Device &device) {
  torch::NoGradGuard guard;
  auto options = packing::bfopts();
  auto integers = torch::arange(4096 * 32, torch::kInt64);
  auto weight = ((integers.remainder(7) - 3).to(torch::kBFloat16) / 64).view({4096, 32});
  std::vector<int64_t> bits{0, 12, 128, 140, 4095};
  auto projected = project(bits, weight, 32);
  auto expected = weight.index_select(0, packing::indices(bits, torch::kCPU)).sum(0);
  auto actual = packing::bf_tensor(projected, 32, torch::kCPU).squeeze(0);
  TORCH_CHECK(torch::equal(actual, expected), "native BF16 bit projection differs");
  Vector x(32), y(32);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = BF((int(i % 5) - 2) / 4.0f);
  auto matrix = weight.narrow(0, 0, 128).view({32, 128}).contiguous();
  auto bias = torch::arange(32, torch::kInt64).remainder(3).to(torch::kBFloat16) / 8;
  linear(x.data(), matrix, bias, 32, 32, y.data(), 32);
  expected = torch::linear(packing::bf_tensor(x, 32, device), matrix.to(device).narrow(1, 32, 32),
                           bias.to(device))
                 .to(torch::kCPU)
                 .squeeze(0);
  actual = packing::bf_tensor(y, 32, torch::kCPU).squeeze(0);
  TORCH_CHECK(torch::equal(actual, expected), "native BF16 strided affine differs");
  for (int64_t count : {1, 3, 4, 5, 12}) {
    Vector batch(count * 32), result(count * 32), reference(count * 32);
    for (size_t i = 0; i < batch.size(); ++i)
      batch[i] = BF((int(i % 17) - 8) / 64.0f);
    auto square = matrix.narrow(1, 0, 32).contiguous();
    linear_rows(batch, square, bias, result, count);
    for (int64_t i = 0; i < count; ++i)
      linear(batch.data() + i * 32, square, bias, 0, 32, reference.data() + i * 32, 32);
    TORCH_CHECK(result == reference, "native candidate tiling changed BF16 reduction order");
  }
  Vector logits{BF(-32), BF(-2), BF(-0.5f), BF(0), BF(1), BF(9), BF(-100), BF(3)};
  std::vector<uint8_t> mask{1, 1, 0, 1, 1, 1, 1, 0};
  auto mass = normalize(logits, mask);
  std::vector<int64_t> owner(logits.size(), 0), valid(mask.begin(), mask.end());
  expected = policy_mass(packing::bf_tensor(logits, logits.size(), device).squeeze(0),
                         packing::indices(owner, device),
                         packing::indices({int64_t(logits.size())}, device),
                         packing::indices(valid, device).ne(0))
                 .to(torch::kCPU);
  for (size_t i = 0; i < mass.size(); ++i)
    TORCH_CHECK(mass[i] == expected.data_ptr<BF>()[i].x,
                "native BF16 categorical definition differs");
}
} // namespace hj
