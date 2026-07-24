// Decode-only SM89 W2 prototype for the six-field descriptor ABI:
//   {activation_fp8, activation_scale_f32, weight_u2, weight_scale_ue8m0,
//    output_bf16, m_rows}
//
// Launch contract:
//   grid  = (N / 64, descriptor_count, 1)
//   block = (64, 1, 1)
//   M must be in [1, 4].
//
// TODO(validation): Run kernels/gen/moe_w2_sm89_native_check.py on an idle
// SM89 GPU before integrating this cubin into the runtime.
// TODO(perf): Compare this N32-per-warp shape with an N16-per-warp variant and
// replace general FP32 scale multiplication with the production exponent-add
// fast path if scale arithmetic is visible in the profile.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr uint32_t kW2ToE4M3 = 0x4838b8c8U;

struct Accumulator {
  float v0;
  float v1;
  float v2;
  float v3;
};

__device__ __forceinline__ uint32_t decode_four_w2(uint32_t packed) {
  // Convert four adjacent 2-bit codes into four E4M3 bytes:
  //   0 -> -4, 1 -> -1, 2 -> +1, 3 -> +4.
  const uint32_t selector =
      (packed & 0x03U) | ((packed & 0x0cU) << 2) |
      ((packed & 0x30U) << 4) | ((packed & 0xc0U) << 6);
  uint32_t decoded;
  const uint32_t zero = 0;
  asm("prmt.b32 %0, %1, %2, %3;"
      : "=r"(decoded)
      : "r"(kW2ToE4M3), "r"(zero), "r"(selector));
  return decoded;
}

__device__ __forceinline__ float ue8m0_to_float(uint32_t scale) {
  // UE8M0 0 denotes 2^-127, represented as an FP32 subnormal.
  const uint32_t bits = scale == 0 ? 0x00400000U : scale << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ Accumulator mma_m16n8k32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  Accumulator out{0.0F, 0.0F, 0.0F, 0.0F};
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(out.v0), "+f"(out.v1), "+f"(out.v2), "+f"(out.v3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  return out;
#else
#error "moe_w2_sm89_decode.cu must be compiled for SM89 or newer"
#endif
}

__device__ __forceinline__ void decode_weight_fragment(
    uint64_t packed, int k32_half, uint32_t (&fragment)[4]) {
  const int byte_base = k32_half * 2;
  fragment[0] = decode_four_w2(
      static_cast<uint32_t>(packed >> (8 * byte_base)));
  fragment[1] = decode_four_w2(
      static_cast<uint32_t>(packed >> (8 * (byte_base + 4))));
  fragment[2] = decode_four_w2(
      static_cast<uint32_t>(packed >> (8 * (byte_base + 1))));
  fragment[3] = decode_four_w2(
      static_cast<uint32_t>(packed >> (8 * (byte_base + 5))));
}

template <int K>
__device__ __forceinline__ void apply_scales(
    Accumulator &sum, const Accumulator &partial, const float *activation_scale,
    const uint8_t *weight_scale, int n_block16, int k_group32, int m_rows,
    int group_id, int thread_id) {
  if (thread_id >= 2) {
    return;
  }

  constexpr int kGroups32 = K / 32;
  const int token0 = thread_id * 2;
  const int token1 = token0 + 1;
  const uint8_t *weight_group =
      weight_scale + (n_block16 * kGroups32 + k_group32) * 16;
  const float row0_scale = ue8m0_to_float(weight_group[group_id]);
  const float row8_scale = ue8m0_to_float(weight_group[group_id + 8]);

  if (token0 < m_rows) {
    const float activation =
        activation_scale[token0 * kGroups32 + k_group32];
    sum.v0 = fmaf(partial.v0, activation * row0_scale, sum.v0);
    sum.v2 = fmaf(partial.v2, activation * row8_scale, sum.v2);
  }
  if (token1 < m_rows) {
    const float activation =
        activation_scale[token1 * kGroups32 + k_group32];
    sum.v1 = fmaf(partial.v1, activation * row0_scale, sum.v1);
    sum.v3 = fmaf(partial.v3, activation * row8_scale, sum.v3);
  }
}

template <int N>
__device__ __forceinline__ void store_tile(
    __nv_bfloat16 *output, const Accumulator &sum, int n_block16, int m_rows,
    int group_id, int thread_id) {
  if (thread_id >= 2) {
    return;
  }

  const int token0 = thread_id * 2;
  const int token1 = token0 + 1;
  const int row0 = n_block16 * 16 + group_id;
  const int row8 = row0 + 8;
  if (token0 < m_rows) {
    output[token0 * N + row0] = __float2bfloat16_rn(sum.v0);
    output[token0 * N + row8] = __float2bfloat16_rn(sum.v2);
  }
  if (token1 < m_rows) {
    output[token1 * N + row0] = __float2bfloat16_rn(sum.v1);
    output[token1 * N + row8] = __float2bfloat16_rn(sum.v3);
  }
}

template <int K, int N>
__device__ __forceinline__ void decode_kernel(const uint64_t *descriptors) {
  static_assert(K % 64 == 0, "K must be a multiple of 64");
  static_assert(N % 64 == 0, "N must be a multiple of 64");

  const uint64_t *descriptor = descriptors + blockIdx.y * 6;
  const int m_rows = static_cast<int>(descriptor[5]);
  if (m_rows < 1 || m_rows > 4) {
    return;
  }

  const auto *activation =
      reinterpret_cast<const uint8_t *>(descriptor[0]);
  const auto *activation_scale =
      reinterpret_cast<const float *>(descriptor[1]);
  const auto *weight = reinterpret_cast<const uint8_t *>(descriptor[2]);
  const auto *weight_scale =
      reinterpret_cast<const uint8_t *>(descriptor[3]);
  auto *output = reinterpret_cast<__nv_bfloat16 *>(descriptor[4]);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group_id = lane >> 2;
  const int thread_id = lane & 3;
  const int n_block16_0 = blockIdx.x * 4 + warp * 2;
  const int n_block16_1 = n_block16_0 + 1;

  constexpr int kBlocks64 = K / 64;
  Accumulator sum0{0.0F, 0.0F, 0.0F, 0.0F};
  Accumulator sum1{0.0F, 0.0F, 0.0F, 0.0F};

#pragma unroll 1
  for (int k_block64 = 0; k_block64 < kBlocks64; ++k_block64) {
    const auto *weight64 = reinterpret_cast<const uint64_t *>(weight);
    const uint64_t packed0 =
        weight64[(n_block16_0 * kBlocks64 + k_block64) * 32 + lane];
    const uint64_t packed1 =
        weight64[(n_block16_1 * kBlocks64 + k_block64) * 32 + lane];

#pragma unroll
    for (int k32_half = 0; k32_half < 2; ++k32_half) {
      const int k_group32 = k_block64 * 2 + k32_half;
      uint32_t activation_fragment0 = 0;
      uint32_t activation_fragment1 = 0;
      if (group_id < m_rows) {
        const uint8_t *activation_group =
            activation + group_id * K + k_group32 * 32 + thread_id * 4;
        activation_fragment0 =
            *reinterpret_cast<const uint32_t *>(activation_group);
        activation_fragment1 =
            *reinterpret_cast<const uint32_t *>(activation_group + 16);
      }

      uint32_t weight_fragment0[4];
      uint32_t weight_fragment1[4];
      decode_weight_fragment(packed0, k32_half, weight_fragment0);
      decode_weight_fragment(packed1, k32_half, weight_fragment1);

      const Accumulator partial0 = mma_m16n8k32(
          weight_fragment0[0], weight_fragment0[1], weight_fragment0[2],
          weight_fragment0[3], activation_fragment0, activation_fragment1);
      const Accumulator partial1 = mma_m16n8k32(
          weight_fragment1[0], weight_fragment1[1], weight_fragment1[2],
          weight_fragment1[3], activation_fragment0, activation_fragment1);

      apply_scales<K>(sum0, partial0, activation_scale, weight_scale,
                      n_block16_0, k_group32, m_rows, group_id, thread_id);
      apply_scales<K>(sum1, partial1, activation_scale, weight_scale,
                      n_block16_1, k_group32, m_rows, group_id, thread_id);
    }
  }

  store_tile<N>(output, sum0, n_block16_0, m_rows, group_id, thread_id);
  store_tile<N>(output, sum1, n_block16_1, m_rows, group_id, thread_id);
}

}  // namespace

extern "C" __global__ __launch_bounds__(64, 8)
void moe_w2_sm89_decode_k1024_n4096(const uint64_t *descriptors) {
  decode_kernel<1024, 4096>(descriptors);
}

extern "C" __global__ __launch_bounds__(64, 8)
void moe_w2_sm89_decode_k4096_n2048(const uint64_t *descriptors) {
  decode_kernel<4096, 2048>(descriptors);
}
