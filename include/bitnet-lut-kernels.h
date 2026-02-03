#if defined(GGML_BITNET_ARM_TL1)
#ifdef __cplusplus
#include <cstdio>
#include <cstring>
#else
#include <stdio.h>
#include <string.h>
#endif
#include <arm_neon.h>
#include "ggml-bitnet.h"

#define GGML_BITNET_MAX_NODES 8192
#define BM 128
#define BK 64

#ifdef  __cplusplus
extern "C" {
#endif

extern bool initialized;
extern bitnet_tensor_extra * bitnet_tensor_extras;
extern size_t bitnet_tensor_extras_index;

extern void * aligned_malloc(size_t size);
extern void aligned_free(void * ptr);
extern bool is_type_supported(enum ggml_type type);

#ifdef  __cplusplus
}
#endif

// Inline helpers
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
{
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
}

inline void reconstruct_int16_pair(int8x16_t high, int8x16_t low, int16x8_t& out_lo, int16x8_t& out_hi) {

    int16x8_t high_lo = vshlq_n_s16(vmovl_s8(vget_low_s8(high)), 8);
    int16x8_t high_hi = vshlq_n_s16(vmovl_s8(vget_high_s8(high)), 8);
    int16x8_t low_lo = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_low_s8(low))));
    int16x8_t low_hi = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_high_s8(low))));
    out_lo = vorrq_s16(high_lo, low_lo);
    out_hi = vorrq_s16(high_hi, low_hi);

}
#endif
