#if defined(GGML_BITNET_ARM_TL1)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include <cmath>
#include "ggml-bitnet.h"
#include "bitnet-lut-kernels.h"

#ifndef __ARM_NEON
#error "__ARM_NEON is not defined - NEON optimizations will not be used. Check compiler flags: -march=armv8-a"
#endif

#define BM 128
#define BK 64

bool initialized = false;
bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
size_t bitnet_tensor_extras_index = 0;

void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}

void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void per_tensor_quant(int k, void* lut_scales_, void* b_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    bitnet_float_type* b = (bitnet_float_type*)b_;
#ifdef __ARM_NEON
    float32x4_t temp_max = vdupq_n_f32(0);
    for (int i=0; i < k / 4; i++) {
      float32x4_t vec_bs = vld1q_f32(b + 4 * i);
      float32x4_t abssum = vabsq_f32(vec_bs);
      temp_max = vmaxq_f32(abssum, temp_max);
    }
    float32_t max_val = vmaxvq_f32(temp_max);
    *lut_scales = (max_val > 0) ? (127.0f / max_val) : 1.0f;
#elif defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k / 8; i++) {
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float max_val = _mm_cvtss_f32(max1);
    *lut_scales = (max_val > 0) ? (127.0f / max_val) : 1.0f;
#else
    // Fallback: scalar implementation
    float max_val = 0.0f;
    for (int i = 0; i < k; i++) {
        float abs_val = fabs(b[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    *lut_scales = (max_val > 0) ? (127.0f / max_val) : 1.0f;
#endif
}

static void partial_max_reset(void* lut_scales_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    *lut_scales = 0.0;
}

bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_TL1) {
        return true;
    } else {
        return false;
    }
}

void ggml_preprocessor(int M, int K, void* B, void* LUT_Scales, void* QLUT) {
  partial_max_reset((&(((bitnet_float_type*)LUT_Scales)[0])));
  per_tensor_quant(K, (&(((bitnet_float_type*)LUT_Scales)[0])), (&(((bitnet_float_type*)B)[0])));
  
  lut_ctor(K, (&(((int8_t*)QLUT)[0])), (&(((bitnet_float_type*)B)[0])), (&(((bitnet_float_type*)LUT_Scales)[0])));
}

void ggml_qgemm_lut(int M, int N, int K, int ii, int j, uint8_t* A, int8_t* LUT, void* Scales, void* LUT_Scales, float32_t* C) {
    int KK = K / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    
    // Pre-allocate LUT arrays outside loops to reduce stack pressure
    int8x16_t* vec_lut_high = (int8x16_t*)aligned_malloc(BK * sizeof(int8x16_t));
    int8x16_t* vec_lut_low = (int8x16_t*)aligned_malloc(BK * sizeof(int8x16_t));

    for (int kk = 0; kk < KK; kk += BK) {        
        // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
        for (int k = 0; k < BK; k++) {
            vec_lut_high[k] = vld1q_s8(LUT + (kk + k) * 32);      // Load high bytes
            vec_lut_low[k] = vld1q_s8(LUT + (kk + k) * 32 + 16);   // Load low bytes
        }
        
#pragma unroll
        for (int i = ii; i < ii + BM; i += 32) {
            int16x8_t vec_c[4] = {vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0)};
            int i_div2 = i / 2;  // Compute once instead of BK times per iteration
#pragma unroll
            for (int k = kk; k < kk + BK; k++) {
                uint8x16_t vec_a = vld1q_u8(A + k * M / 2 + i_div2);
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

            float32_t* pC = (float32_t*) &(C[(i+0)*N + j]);
            const float32_t lut_scale = ((float32_t*)LUT_Scales)[0];
            const float32_t scale = ((float32_t*)Scales)[0];
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
    
    // Cleanup heap allocations
    aligned_free(vec_lut_high);
    aligned_free(vec_lut_low);
}

void ggml_vec_dot_tl1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc)
{
    GGML_ASSERT(false); // I don't this needs to be implemented for BitNet TL1

    /*int KK = K / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);

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
            int i_div2 = i / 2;  // Compute once instead of BK times per iteration
#pragma unroll
            for (int k = kk; k < kk + BK; k++) {
                uint8x16_t vec_a = vld1q_u8(A + k * M / 2 + i_div2);
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

            float32_t* pC = (float32_t*) &(C[(i+0)*N + j]);
            const float32_t lut_scale = ((float32_t*)LUT_Scales)[0];
            const float32_t scale = ((float32_t*)Scales)[0];
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
    }*/
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
