#if defined(GGML_BITNET_ARM_TL1)
#include "ggml-bitnet.h"
#include <cstdio>
#include <cstring>
#define GGML_BITNET_MAX_NODES 8192
#define BM 128
#define BK 64

static bool initialized = false;
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
static size_t bitnet_tensor_extras_index = 0;
static void * aligned_malloc(size_t size) {{
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}}
static void aligned_free(void * ptr) {{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}}

#include <arm_neon.h>

static void per_tensor_quant(int k, void* lut_scales_, void* b_) {{
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    bitnet_float_type* b = (bitnet_float_type*)b_;
#ifdef __ARM_NEON
    float32x4_t temp_max = vdupq_n_f32(0);
    for (int i=0; i < k / 4; i++) {{
      float32x4_t vec_bs = vld1q_f32(b + 4 * i);
      float32x4_t abssum = vabsq_f32(vec_bs);
      temp_max = vmaxq_f32(abssum, temp_max);
    }}
    float32_t scales = 127 / vmaxvq_f32(temp_max);
    *lut_scales = scales;
#elif defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    // #pragma unroll
    for (int i = 0; i < k / 8; i++) {{
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }}
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float scales = 127 / _mm_cvtss_f32(max1);
    *lut_scales = scales;
#endif
}}

static void partial_max_reset(void* lut_scales_) {{
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    *lut_scales = 0.0;
}}

#ifdef __ARM_NEON
inline void Transpose_8_8(
    int16x8_t *v0,
    int16x8_t *v1,
    int16x8_t *v2,
    int16x8_t *v3,
    int16x8_t *v4,
    int16x8_t *v5,
    int16x8_t *v6,
    int16x8_t *v7)
{{
    int16x8x2_t q04 = vzipq_s16(*v0, *v4);
    int16x8x2_t q15 = vzipq_s16(*v1, *v5);
    int16x8x2_t q26 = vzipq_s16(*v2, *v6);
    int16x8x2_t q37 = vzipq_s16(*v3, *v7);

    int16x8x2_t q0246_0 = vzipq_s16(q04.val[0], q26.val[0]);
    int16x8x2_t q0246_1 = vzipq_s16(q04.val[1], q26.val[1]);
    int16x8x2_t q1357_0 = vzipq_s16(q15.val[0], q37.val[0]);
    int16x8x2_t q1357_1 = vzipq_s16(q15.val[1], q37.val[1]);

    int16x8x2_t q_fin_0 = vzipq_s16(q0246_0.val[0], q1357_0.val[0]);
    int16x8x2_t q_fin_1 = vzipq_s16(q0246_0.val[1], q1357_0.val[1]);
    int16x8x2_t q_fin_2 = vzipq_s16(q0246_1.val[0], q1357_1.val[0]);
    int16x8x2_t q_fin_3 = vzipq_s16(q0246_1.val[1], q1357_1.val[1]);

    *v0 = q_fin_0.val[0];
    *v1 = q_fin_0.val[1];
    *v2 = q_fin_1.val[0];
    *v3 = q_fin_1.val[1];
    *v4 = q_fin_2.val[0];
    *v5 = q_fin_2.val[1];
    *v6 = q_fin_3.val[0];
    *v7 = q_fin_3.val[1];
}}
#endif

inline void lut_ctor(int act_k, int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {{
#ifdef __ARM_NEON
    int16x8_t vec_lut[16];
    float32_t scales = *lut_scales;
    uint8_t tbl_mask[16];
    tbl_mask[0] = 0;
    tbl_mask[1] = 2;
    tbl_mask[2] = 4;
    tbl_mask[3] = 6;
    tbl_mask[4] = 8;
    tbl_mask[5] = 10;
    tbl_mask[6] = 12;
    tbl_mask[7] = 14;
    tbl_mask[8] = 1;
    tbl_mask[9] = 3;
    tbl_mask[10] = 5;
    tbl_mask[11] = 7;
    tbl_mask[12] = 9;
    tbl_mask[13] = 11;
    tbl_mask[14] = 13;
    tbl_mask[15] = 15;
    uint8x16_t tbl_mask_q = vld1q_u8(tbl_mask);
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {{
        float32x4x2_t vec_bs_x0 = vld2q_f32(b + k * 16);
        float32x4x2_t vec_bs_x1 = vld2q_f32(b + k * 16 + 8);
        float32x4_t vec_f_0 = vmulq_n_f32(vec_bs_x0.val[0], scales);
        float32x4_t vec_f_1 = vmulq_n_f32(vec_bs_x0.val[1], scales);
        float32x4_t vec_f_2 = vmulq_n_f32(vec_bs_x1.val[0], scales);
        float32x4_t vec_f_3 = vmulq_n_f32(vec_bs_x1.val[1], scales);
        int32x4_t vec_b_0 = vcvtnq_s32_f32(vec_f_0);
        int32x4_t vec_b_1 = vcvtnq_s32_f32(vec_f_1);
        int32x4_t vec_b_2 = vcvtnq_s32_f32(vec_f_2);
        int32x4_t vec_b_3 = vcvtnq_s32_f32(vec_f_3);
        int16x4_t vec_b16_0 = vmovn_s32(vec_b_0);
        int16x4_t vec_b16_1 = vmovn_s32(vec_b_1);
        int16x4_t vec_b16_2 = vmovn_s32(vec_b_2);
        int16x4_t vec_b16_3 = vmovn_s32(vec_b_3);
        int16x8_t vec_bs_0 = vcombine_s16(vec_b16_0, vec_b16_2);
        int16x8_t vec_bs_1 = vcombine_s16(vec_b16_1, vec_b16_3);
        vec_lut[0] = vdupq_n_s16(0);
        vec_lut[0] = vec_lut[0] - vec_bs_0;
        vec_lut[0] = vec_lut[0] - vec_bs_1;
        vec_lut[1] = vdupq_n_s16(0);
        vec_lut[1] = vec_lut[1] - vec_bs_0;
        vec_lut[2] = vdupq_n_s16(0);
        vec_lut[2] = vec_lut[2] - vec_bs_0;
        vec_lut[2] = vec_lut[2] + vec_bs_1;
        vec_lut[3] = vdupq_n_s16(0);
        vec_lut[3] = vec_lut[3] - vec_bs_1;
        vec_lut[4] = vdupq_n_s16(0);
        vec_lut[5] = vec_bs_1;
        vec_lut[6] = vec_bs_0;
        vec_lut[6] = vec_lut[6] - vec_bs_1;
        vec_lut[7] = vec_bs_0;
        vec_lut[8] = vec_bs_0;
        vec_lut[8] = vec_lut[8] + vec_bs_1;
        Transpose_8_8(&(vec_lut[0]), &(vec_lut[1]), &(vec_lut[2]), &(vec_lut[3]),
                      &(vec_lut[4]), &(vec_lut[5]), &(vec_lut[6]), &(vec_lut[7]));
        Transpose_8_8(&(vec_lut[8]), &(vec_lut[9]), &(vec_lut[10]), &(vec_lut[11]),
                      &(vec_lut[12]), &(vec_lut[13]), &(vec_lut[14]), &(vec_lut[15]));
#pragma unroll
        for (int idx = 0; idx < 8; idx++) {{
            int8x16_t q0_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx]), tbl_mask_q);
            int8x8_t q0_low = vget_low_s8(q0_s);
            int8x8_t q0_high = vget_high_s8(q0_s);
            int8x16_t q1_s = vqtbl1q_s8(vreinterpretq_s8_s16(vec_lut[idx + 8]), tbl_mask_q);
            int8x8_t q1_low = vget_low_s8(q1_s);
            int8x8_t q1_high = vget_high_s8(q1_s);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2, q0_high);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 8, q1_high);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 16, q0_low);
            vst1_s8(qlut + k * 16 * 8 * 2 + idx * 16 * 2 + 24, q1_low);
        }}
    }}
#endif
}}

static bool is_type_supported(enum ggml_type type) {{
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_TL1) {{
        return true;
    }} else {{
        return false;
    }}
}}

void ggml_preprocessor(int M, int K, void* B, void* LUT_Scales, void* QLUT) {
  partial_max_reset((&(((bitnet_float_type*)LUT_Scales)[0])));
  per_tensor_quant(K, (&(((bitnet_float_type*)LUT_Scales)[0])), (&(((bitnet_float_type*)B)[0])));
  
  lut_ctor(K, (&(((int8_t*)QLUT)[0])), (&(((bitnet_float_type*)B)[0])), (&(((bitnet_float_type*)LUT_Scales)[0])));
}

inline void reconstruct_int16_pair(int8x16_t high, int8x16_t low, int16x8_t& out_lo, int16x8_t& out_hi) {
    int16x8_t high_lo = vshlq_n_s16(vmovl_s8(vget_low_s8(high)), 8);
    int16x8_t high_hi = vshlq_n_s16(vmovl_s8(vget_high_s8(high)), 8);
    int16x8_t low_lo = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_low_s8(low))));
    int16x8_t low_hi = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_high_s8(low))));
    out_lo = vorrq_s16(high_lo, low_lo);
    out_hi = vorrq_s16(high_hi, low_hi);
}

void ggml_qgemm_lut(int M, int N, int K, int ii, int j, uint8_t* A, int8_t* LUT, void* Scales, void* LUT_Scales, float32_t* C) {
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const int KK = K/2;

    for (int kk = 0; kk < KK; kk += BK) {
        int8x16_t vec_lut_high[BK];
        int8x16_t vec_lut_low[BK];
        
        // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
        for (int k = 0; k < BK; k++) {
            vec_lut_high[k] = vld1q_s8(LUT + (kk + k) * 32);      // Load high bytes
            vec_lut_low[k] = vld1q_s8(LUT + (kk + k) * 32 + 16);   // Load low bytes
        }
        
#pragma unroll
        for (int i = ii; i < ii + BM; i += 32) {
            int16x8_t vec_c[4] = {vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0)};
#pragma unroll
            for (int k = kk; k < kk + BK; k++) {
                // Load 16 activations from same k, different rows (from transposed A)
                uint8x16_t vec_a = vld1q_u8(A + k * M + i);
                uint8x16_t vec_a_top = vshrq_n_u8(vec_a, 4);
                uint8x16_t vec_a_bot = vandq_u8(vec_a, vec_mask);
                uint8x16x2_t vec_a_unpacked = vzipq_u8(vec_a_top, vec_a_bot);

                // Lookup on high and low tables (same LUT table for all 16 indices)
                int8x16_t vec_l0_h = vqtbl1q_s8(vec_lut_high[k - kk], vec_a_unpacked.val[0]);
                int8x16_t vec_l0_l = vqtbl1q_s8(vec_lut_low[k - kk], vec_a_unpacked.val[0]);

                int16x8_t out0, out1;
                reconstruct_int16_pair(vec_l0_h, vec_l0_l, out0, out1);    
                vec_c[0] = vaddq_s16(vec_c[0], out0);
                vec_c[1] = vaddq_s16(vec_c[1], out1);

                // Lookup on high and low tables (same LUT table for all 16 indices)
                int8x16_t vec_l1_h = vqtbl1q_s8(vec_lut_high[k - kk], vec_a_unpacked.val[1]);
                int8x16_t vec_l1_l = vqtbl1q_s8(vec_lut_low[k - kk], vec_a_unpacked.val[1]);
                
                int16x8_t out2, out3;
                reconstruct_int16_pair(vec_l1_h, vec_l1_l, out2, out3);    
                vec_c[2] = vaddq_s16(vec_c[2], out2);
                vec_c[3] = vaddq_s16(vec_c[3], out3); 
            }

            // Extract and store results
            bitnet_float_type* pC = (bitnet_float_type*) &(C[(i+0)*N + j]);
            const float32_t lut_scale = ((bitnet_float_type*)LUT_Scales)[0];
            const float32_t scale = ((bitnet_float_type*)Scales)[0];
            int16_t tmp_vals[8];
#pragma unroll
            for (int block = 0; block < 4; ++block) {
                vst1q_s16(tmp_vals, vec_c[block]);
                for (int lane = 0; lane < 8; ++lane, pC += N) {
                    float32_t val = (tmp_vals[lane] / lut_scale) * scale;
                    (*pC) += val;
                }
            }
        }
    }
}

void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {
        return;
    }

    int m = tensor->ne[0];
    int k = tensor->ne[1];
    const int lut_scales_size = 1;
    const int scales_size = 1;

    GGML_ASSERT(k==2560 || k==6912 || k == 640);
    GGML_ASSERT(m==2560 || m==640 || m == 6912);

    const int n_tile_num = m / BM;
    uint8_t * qweights;
    bitnet_float_type * scales;

    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));
    qweights = (uint8_t *) tensor->data;
    float * i2_scales = (float * )(qweights + k * m / 4);
    scales[0] = (bitnet_float_type) i2_scales[0];

    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .BK              = */ BK,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };
}
#endif