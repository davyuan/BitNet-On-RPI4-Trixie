#include <vector>
#include <type_traits>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "bitnet-lut-kernels.h"

#if defined(GGML_BITNET_ARM_TL1)

#define epsilon 1e-7f

bool initialized = false;
bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
size_t bitnet_tensor_extras_index = 0;

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

extern "C" {

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

void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

} // extern "C"

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        // Add additional args to decide if permuted I2 or naive I2
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_bitnet_get_type_bits(src0->type);
    
    size_t wsize = ne11 * ne10 * 16 * sizeof(int8_t) + ne11  * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        src0->backend == GGML_BACKEND_TYPE_CPU) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
#endif