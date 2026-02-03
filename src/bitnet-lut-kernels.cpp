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
#define epsilon 1e-7f

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

float ggml_get_tensor_max(int k, void* b_) {
    bitnet_float_type* b = (bitnet_float_type*)b_;
#ifdef __ARM_NEON
    float32x4_t temp_max = vdupq_n_f32(0);
    for (int i=0; i < k / 4; i++) {
      float32x4_t vec_bs = vld1q_f32(b + 4 * i);
      float32x4_t abssum = vabsq_f32(vec_bs);
      temp_max = vmaxq_f32(abssum, temp_max);
    }
    float32_t max_val = vmaxvq_f32(temp_max);
    return max_val;
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
    *lut_scales = (max_val > epsilon) ? (127.0f / max_val) : 1.0f;
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
    *lut_scales = (max_val > epsilon) ? (127.0f / max_val) : 1.0f;
#else
    // Fallback: scalar implementation
    float max_val = 0.0f;
    for (int i = 0; i < k; i++) {
        float abs_val = fabs(b[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    *lut_scales = (max_val > epsilon) ? (127.0f / max_val) : 1.0f;
#endif
}

static void partial_max_reset(void* lut_scales_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    *lut_scales = 0.0;
}

static void ggml_lut_ctor(int act_k, int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {
#ifdef __ARM_NEON
    int16x8_t vec_lut[16];
    // Initialization to avoid uninitialized warnings and zero out any potential garbage
    for (int i = 0; i < 16; i++) vec_lut[i] = vdupq_n_s16(0);

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

    for (int k = 0; k < act_k / 16; ++k) {
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
        vec_lut[0] = vsubq_s16(vec_lut[0], vec_bs_0);
        vec_lut[0] = vsubq_s16(vec_lut[0], vec_bs_1);
        vec_lut[1] = vdupq_n_s16(0);
        vec_lut[1] = vsubq_s16(vec_lut[1], vec_bs_0);
        vec_lut[2] = vdupq_n_s16(0);
        vec_lut[2] = vsubq_s16(vec_lut[2], vec_bs_0);
        vec_lut[2] = vaddq_s16(vec_lut[2], vec_bs_1);
        vec_lut[3] = vdupq_n_s16(0);
        vec_lut[3] = vsubq_s16(vec_lut[3], vec_bs_1);
        vec_lut[4] = vdupq_n_s16(0);
        vec_lut[5] = vec_bs_1;
        vec_lut[6] = vsubq_s16(vec_bs_0, vec_bs_1);
        vec_lut[7] = vec_bs_0;
        vec_lut[8] = vaddq_s16(vec_bs_0, vec_bs_1);
        
        for(int idx = 9; idx < 16; idx++) vec_lut[idx] = vdupq_n_s16(0);

        Transpose_8_8(&(vec_lut[0]), &(vec_lut[1]), &(vec_lut[2]), &(vec_lut[3]),
                      &(vec_lut[4]), &(vec_lut[5]), &(vec_lut[6]), &(vec_lut[7]));
        Transpose_8_8(&(vec_lut[8]), &(vec_lut[9]), &(vec_lut[10]), &(vec_lut[11]),
                      &(vec_lut[12]), &(vec_lut[13]), &(vec_lut[14]), &(vec_lut[15]));

        for (int idx = 0; idx < 8; idx++) {
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
        }
    }
#endif
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

    ggml_lut_ctor(K, (&(((int8_t*)QLUT)[0])), (&(((bitnet_float_type*)B)[0])), (&(((bitnet_float_type*)LUT_Scales)[0])));
}

void ggml_qgemm_lut(int M, int N, int K, int ii, int j, uint8_t* A, int8_t* LUT, void* Scales, void* LUT_Scales, float32_t* C) {
    int KK = K / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const float32_t lut_scale = ((float32_t*)LUT_Scales)[0];
    const float32_t weight_scale = ((float32_t*)Scales)[0];
    const float32x4_t v_rescale = vdupq_n_f32(weight_scale / lut_scale);

    // Initializate accumulators for 128 rows
    int16x8_t acc[16];
    for (int b = 0; b < 16; b++) acc[b] = vdupq_n_s16(0);

    const int i_packed = ii / 2;
    const int row_stride = M / 2;

    for (int k = 0; k < KK; k++) {
        int8x16_t vh = vld1q_s8(LUT + k * 32);
        int8x16_t vl = vld1q_s8(LUT + k * 32 + 16);

#define PROCESS_32_ROWS(a_ptr, accl0, accl1, acch0, acch1) { \
            uint8x16_t vec_a = vld1q_u8(a_ptr); \
            uint8x16_t vec_a_top = vshrq_n_u8(vec_a, 4); \
            uint8x16_t vec_a_bot = vandq_u8(vec_a, vec_mask); \
            uint8x16x2_t vec_a_unpacked = vzipq_u8(vec_a_top, vec_a_bot); \
            int8x16_t r0h = vqtbl1q_s8(vh, vec_a_unpacked.val[0]); \
            int8x16_t r0l = vqtbl1q_s8(vl, vec_a_unpacked.val[0]); \
            int8x16_t r1h = vqtbl1q_s8(vh, vec_a_unpacked.val[1]); \
            int8x16_t r1l = vqtbl1q_s8(vl, vec_a_unpacked.val[1]); \
            int16x8_t o0, o1, o2, o3; \
            reconstruct_int16_pair(r0h, r0l, o0, o1); \
            reconstruct_int16_pair(r1h, r1l, o2, o3); \
            accl0 = vaddq_s16(accl0, o0); \
            accl1 = vaddq_s16(accl1, o1); \
            acch0 = vaddq_s16(acch0, o2); \
            acch1 = vaddq_s16(acch1, o3); \
        }

        const uint8_t* pA = A + k * row_stride + i_packed;
        PROCESS_32_ROWS(pA,      acc[0], acc[1], acc[2], acc[3]);
        PROCESS_32_ROWS(pA + 16, acc[4], acc[5], acc[6], acc[7]);
        PROCESS_32_ROWS(pA + 32, acc[8], acc[9], acc[10], acc[11]);
        PROCESS_32_ROWS(pA + 48, acc[12], acc[13], acc[14], acc[15]);
#undef PROCESS_32_ROWS
    }

    // Write-back
    for (int block = 0; block < 4; block++) {
        float32_t* pC = &(C[j * M + ii + block * 32]);
#define WRITE_BACK(out_ptr, accl, acch) { \
            vst1q_f32(out_ptr + 0, vfmaq_f32(vld1q_f32(out_ptr + 0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(accl))), v_rescale)); \
            vst1q_f32(out_ptr + 4, vfmaq_f32(vld1q_f32(out_ptr + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(accl))), v_rescale)); \
            vst1q_f32(out_ptr + 8, vfmaq_f32(vld1q_f32(out_ptr + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(acch))), v_rescale)); \
            vst1q_f32(out_ptr + 12, vfmaq_f32(vld1q_f32(out_ptr + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(acch))), v_rescale)); \
        }
        WRITE_BACK(pC,      acc[block*4 + 0], acc[block*4 + 1]);
        WRITE_BACK(pC + 16, acc[block*4 + 2], acc[block*4 + 3]);
#undef WRITE_BACK
    }
}

void ggml_qgemm_lut_2col(int M, int N, int K, int ii, int j, uint8_t* A, int8_t* LUT0, int8_t* LUT1, void* Scales, void* LUT_Scales, float32_t* C) {
    int KK = K / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const float32_t lut_scale0 = ((float32_t*)LUT_Scales)[0];
    const float32_t lut_scale1 = ((float32_t*)LUT_Scales)[1];
    const float32_t weight_scale = ((float32_t*)Scales)[0];
    const float32x4_t v_rescale0 = vdupq_n_f32(weight_scale / lut_scale0);
    const float32x4_t v_rescale1 = vdupq_n_f32(weight_scale / lut_scale1);

    // Initializate accumulators for 128 rows, 2 columns
    // We use BM=128, so 4 blocks of 32 rows. 
    // total 8 accs per block (4 per column). Total 32 accs. 
    int16x8_t acc_j0[16];
    int16x8_t acc_j1[16];
    for (int b = 0; b < 16; b++) {
        acc_j0[b] = vdupq_n_s16(0);
        acc_j1[b] = vdupq_n_s16(0);
    }

    const int i_packed = ii / 2;
    const int row_stride = M / 2;

    for (int k = 0; k < KK; k++) {
        int8x16_t vh0 = vld1q_s8(LUT0 + k * 32);
        int8x16_t vl0 = vld1q_s8(LUT0 + k * 32 + 16);
        int8x16_t vh1 = vld1q_s8(LUT1 + k * 32);
        int8x16_t vl1 = vld1q_s8(LUT1 + k * 32 + 16);
        
#define PROCESS_32_ROWS_2COL(a_ptr, idx) { \
            uint8x16_t vec_a = vld1q_u8(a_ptr); \
            uint8x16_t vec_a_top = vshrq_n_u8(vec_a, 4); \
            uint8x16_t vec_a_bot = vandq_u8(vec_a, vec_mask); \
            uint8x16x2_t vec_a_unp = vzipq_u8(vec_a_top, vec_a_bot); \
            \
            /* Col j0 lookups */ \
            int8x16_t r0h_j0 = vqtbl1q_s8(vh0, vec_a_unp.val[0]); \
            int8x16_t r0l_j0 = vqtbl1q_s8(vl0, vec_a_unp.val[0]); \
            int8x16_t r1h_j0 = vqtbl1q_s8(vh0, vec_a_unp.val[1]); \
            int8x16_t r1l_j0 = vqtbl1q_s8(vl0, vec_a_unp.val[1]); \
            /* Col j1 lookups */ \
            int8x16_t r0h_j1 = vqtbl1q_s8(vh1, vec_a_unp.val[0]); \
            int8x16_t r0l_j1 = vqtbl1q_s8(vl1, vec_a_unp.val[0]); \
            int8x16_t r1h_j1 = vqtbl1q_s8(vh1, vec_a_unp.val[1]); \
            int8x16_t r1l_j1 = vqtbl1q_s8(vl1, vec_a_unp.val[1]); \
            \
            int16x8_t o0, o1, o2, o3; \
            reconstruct_int16_pair(r0h_j0, r0l_j0, o0, o1); \
            acc_j0[idx+0] = vaddq_s16(acc_j0[idx+0], o0); acc_j0[idx+1] = vaddq_s16(acc_j0[idx+1], o1); \
            reconstruct_int16_pair(r1h_j0, r1l_j0, o2, o3); \
            acc_j0[idx+2] = vaddq_s16(acc_j0[idx+2], o2); acc_j0[idx+3] = vaddq_s16(acc_j0[idx+3], o3); \
            \
            reconstruct_int16_pair(r0h_j1, r0l_j1, o0, o1); \
            acc_j1[idx+0] = vaddq_s16(acc_j1[idx+0], o0); acc_j1[idx+1] = vaddq_s16(acc_j1[idx+1], o1); \
            reconstruct_int16_pair(r1h_j1, r1l_j1, o2, o3); \
            acc_j1[idx+2] = vaddq_s16(acc_j1[idx+2], o2); acc_j1[idx+3] = vaddq_s16(acc_j1[idx+3], o3); \
        }

        const uint8_t* pA = A + k * row_stride + i_packed;
        PROCESS_32_ROWS_2COL(pA, 0);
        PROCESS_32_ROWS_2COL(pA + 16, 4);
        PROCESS_32_ROWS_2COL(pA + 32, 8);
        PROCESS_32_ROWS_2COL(pA + 48, 12);
#undef PROCESS_32_ROWS_2COL
    }

    // Write-back
    for (int block = 0; block < 4; block++) {
        float32_t* pC0 = &(C[j * M + ii + block * 32]);
        float32_t* pC1 = &(C[(j + 1) * M + ii + block * 32]);

#define WRITE_BACK(out_ptr, accl, acch, rescale) { \
            vst1q_f32(out_ptr + 0, vfmaq_f32(vld1q_f32(out_ptr + 0), vcvtq_f32_s32(vmovl_s16(vget_low_s16(accl))), rescale)); \
            vst1q_f32(out_ptr + 4, vfmaq_f32(vld1q_f32(out_ptr + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(accl))), rescale)); \
            vst1q_f32(out_ptr + 8, vfmaq_f32(vld1q_f32(out_ptr + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(acch))), rescale)); \
            vst1q_f32(out_ptr + 12, vfmaq_f32(vld1q_f32(out_ptr + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(acch))), rescale)); \
        }

        WRITE_BACK(pC0, acc_j0[block*4 + 0], acc_j0[block*4 + 1], v_rescale0);
        WRITE_BACK(pC0 + 16, acc_j0[block*4 + 2], acc_j0[block*4 + 3], v_rescale0);
        WRITE_BACK(pC1, acc_j1[block*4 + 0], acc_j1[block*4 + 1], v_rescale1);
        WRITE_BACK(pC1 + 16, acc_j1[block*4 + 2], acc_j1[block*4 + 3], v_rescale1);
#undef WRITE_BACK
    }
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
