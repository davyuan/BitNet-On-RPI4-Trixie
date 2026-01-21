#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <arm_neon.h>
#include <random>
#include "ggml-bitnet.h"
#include "bitnet-lut-kernels.h"

#define TILE_K 32
#define TILE_N 16
#define TILE_M 4
#define TILE_SIZE 2

const int M =2560;           // Weight rows (A rows)
const int K = 2560;        // Shared dimension
const int N = 160;         // Activation rows (B rows) = output size

// Transpose matrix B from (N x M) to B_T (M x N)
void transpose_matrix(float32_t* B, float32_t* B_T, int M, int N) {
    for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                B_T[i * N + j] = B[j * M + i];
        }
    }
}

// Transpose matrix A from (N x M) to A_T (M x N)
void transpose_matrix(uint8_t* A, uint8_t* A_T, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A_T[i * N + j] = A[j * M + i];
        }
    }
}

static inline void interleave_vec_c_block(int16x4_t c0, int16x4_t c1, int16x4_t c2, int16x4_t c3, int32x4_t out[4]) {
    int16x4x2_t zip01 = vzip_s16(c0, c1);
    int16x4x2_t zip23 = vzip_s16(c2, c3);
    int16x4x2_t trn0 = vtrn_s16(zip01.val[0], zip23.val[0]);
    int16x4x2_t trn1 = vtrn_s16(zip01.val[1], zip23.val[1]);
    int16x4x2_t rows01 = vzip_s16(trn0.val[0], trn0.val[1]);
    int16x4x2_t rows23 = vzip_s16(trn1.val[0], trn1.val[1]);

    out[0] = vmovl_s16(rows01.val[0]);
    out[1] = vmovl_s16(rows01.val[1]);
    out[2] = vmovl_s16(rows23.val[0]);
    out[3] = vmovl_s16(rows23.val[1]);
}

void matmul_naive(int8_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (float32_t)A[i*K + k] * (float32_t)B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

/* A(MxK/2), B(NxK)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version natively implements the LUT-based matmul without SIMD optimizations.   
*/
void matmul_lut_naive(int8_t* A, float32_t* B, int32_t* C, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 1.0f;

    // Debug: Print full A matrix
    /*printf("\n=== DEBUG: Full A matrix (M=%d, KK=%d) ===\n", M, KK);
    for (int i = 0; i < M; i++) {
        printf("A[%2d]: ", i);
        for (int k = 0; k < KK; k++) {
            printf("%2d ", (int)A[i*KK + k]);
        }
        printf("\n");
    }

    // Debug: Print full B matrix (first row only for sanity)
    printf("\n=== DEBUG: Full B matrix (N=%d, K=%d) ===\n", N, K);
    for (int i = 0; i < N; i++) {
        printf("B[%2d]: ", i);
        for (int k = 0; k < K; k++) {
            printf("%3.0f ", B[i*K + k]);
        }
        printf("\n");
    }

    // Debug counter for first few iterations
    int debug_count = 0;*/

    // Partition rows among 4 cores
    #pragma omp parallel for num_threads(4) 
    for (int ii = 0; ii < M; ii += TILE_SIZE) {          
        for (int jj = 0; jj < N; jj += TILE_SIZE) {      
            for (int kk = 0; kk < KK; kk += TILE_SIZE) {                
                for (int i = ii; i < ii + TILE_SIZE; i++) {
                    for (int j = jj; j < jj + TILE_SIZE; j++) {                        
                        lut_ctor(K, QLUT, (float32_t*)(B + j* K), LUT_Scales);    
                        
                        // Debug: Print QLUT after construction (first iteration only)
                        /*if (debug_count == 0) {
                            printf("\n=== DEBUG: QLUT values for %2dth row in B ===\n", j);
                            for (int idx = 0; idx < K * 16; idx++) {
                                if ((idx) % 16 == 0) {
                                    printf("\nLUT[%2d]: ", idx/32);
                                }
                                printf("%4d ", (int)QLUT[idx]);
                            }
                            printf("\n");
                        }*/

                        int32_t local_sum = 0; 
                        
                        for (int k = kk; k < kk + TILE_SIZE; k++) {
                            int lut_index = A[i*KK + k];
                            uint8_t high_byte = (uint8_t)QLUT[k * 32 + lut_index];
                            uint8_t low_byte = (uint8_t)QLUT[k * 32 + 16 + lut_index];
                            // Combine as unsigned first, then cast to signed int16
                            int16_t combined = (int16_t)(((uint16_t)high_byte << 8) | (uint16_t)low_byte);
                            
                            local_sum += (int32_t)combined;
                            /*if (debug_count < 64) {
                                printf("DEBUG [%d]: i=%d, j=%d, k=%d, lut_index=%d, high_byte=%u, low_byte=%u, combined=%d, sum=%d\n",
                                       debug_count, i, j, k, lut_index, (unsigned)high_byte, (unsigned)low_byte, (int)combined, (int)local_sum);
                                debug_count++;
                            }*/                            
                        }

                        // Add to result (C is pre-initialized to 0)
                        C[i*N + j] += local_sum;
                    }
                }
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

/* A(MxK/2), B(NxK)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_naive2(int8_t* A, float32_t* B, int32_t* C, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 1.0f;

    for (int j = 0; j < N; j++) {                        
        lut_ctor(K, QLUT, (float32_t*)(B + j* K), LUT_Scales);    
        
        #pragma omp parallel for num_threads(4)
        for (int ii = 0; ii < M; ii += BM) {          
            for (int kk = 0; kk < KK; kk += BK) {                
                for (int i = ii; i < ii + BM; i++) {
                    int32_t local_sum = 0; 
                    
                    for (int k = kk; k < kk + BK; k++) {
                        int lut_index = A[i*KK + k];
                        uint8_t high_byte = (uint8_t)QLUT[k * 32 + lut_index];
                        uint8_t low_byte = (uint8_t)QLUT[k * 32 + 16 + lut_index];
                        // Combine as unsigned first, then cast to signed int16
                        int16_t combined = (int16_t)(((uint16_t)high_byte << 8) | (uint16_t)low_byte);
                        
                        local_sum += (int32_t)combined;                          
                    }

                    // Add to result (C is pre-initialized to 0)
                    C[i*N + j] += local_sum;
                }
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

/* A(K/2 x M), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_simd(uint8_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 2.0f;

    for (int j = 0; j < N; j++) {
        ggml_preprocessor(M, K, (void*)(B + j * K), (void*)LUT_Scales, (void*)QLUT);                  
        //printf("LUT constructed for row %d, scale=%.2f\n", j, *LUT_Scales);    
        
        // Parallelize over row blocks
        #pragma omp parallel for num_threads(4)
        for (int ii = 0; ii < M; ii += BM) {          
            for (int kk = 0; kk < KK; kk += BK) {
                int8x16_t vec_lut_high[BK];
                int8x16_t vec_lut_low[BK];
                
                // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
                for (int k = 0; k < BK; k++) {
                    vec_lut_high[k] = vld1q_s8(QLUT + (kk + k) * 32);      // Load high bytes
                    vec_lut_low[k] = vld1q_s8(QLUT + (kk + k) * 32 + 16);   // Load low bytes
                }
                
#pragma unroll
                for (int i = ii; i < ii + BM; i += 16) {
                    int16x8_t vec_c[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
#pragma unroll
                    for (int k = kk; k < kk + BK; k++) {
                        // Load 16 weights from same k, different rows (from transposed A)
                        uint8x16_t vec_a0 = vld1q_u8(A + k * M + i);
                        
                        // Lookup on high and low tables (same LUT table for all 16 indices)
                        int8x16_t vec_c0_h = vqtbl1q_s8(vec_lut_high[k - kk], vec_a0);
                        int8x16_t vec_c0_l = vqtbl1q_s8(vec_lut_low[k - kk], vec_a0);
                                               
                        // Reconstruct int16 from high/low bytes: (high << 8) | low
                        int16x8_t v0h_lo_16 = vshlq_n_s16(vmovl_s8(vget_low_s8(vec_c0_h)), 8);
                        int16x8_t v0h_hi_16 = vshlq_n_s16(vmovl_s8(vget_high_s8(vec_c0_h)), 8);  
                        int16x8_t v0l_lo_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_low_s8(vec_c0_l))));
                        int16x8_t v0l_hi_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_high_s8(vec_c0_l))));
                        int16x8_t out00 = vorrq_s16(v0h_lo_16, v0l_lo_16);
                        int16x8_t out01 = vorrq_s16(v0h_hi_16, v0l_hi_16);
                        
                        vec_c[0] = vaddq_s16(vec_c[0], out00);
                        vec_c[1] = vaddq_s16(vec_c[1], out01);
                    }

                    float32_t* pC = (float32_t*) &(C[(i+0)*N + j]);
                    const float32_t lut_scale = ((float32_t*)LUT_Scales)[0];
                    const float32_t scale = ((float32_t*)Scales)[0];
                    int16_t tmp_vals[8];
#pragma unroll
                    for (int block = 0; block < 2; ++block) {
                        vst1q_s16(tmp_vals, vec_c[block]);
                        for (int lane = 0; lane < 8; ++lane, pC += N) {
                            float32_t val = (tmp_vals[lane] / lut_scale) * scale;
                            (*pC) += val;
                        }
                    }
                }
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

/* A(K/2 x M), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_simd2(uint8_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT0 = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));
    int8_t* QLUT1 = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));
    int8_t* QLUT2 = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));
    int8_t* QLUT3 = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc( 4 * sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;

    for (int j = 0; j < N; j+=4) {                      
        ggml_preprocessor(M, K, (void*)(B + j * K), (void*)(&(LUT_Scales[0])), (void*)QLUT0);                  
        ggml_preprocessor(M, K, (void*)(B + (j+1) * K), (void*)(&(LUT_Scales[1])), (void*)QLUT1);                  
        ggml_preprocessor(M, K, (void*)(B + (j+2) * K), (void*)(&(LUT_Scales[2])), (void*)QLUT2);                  
        ggml_preprocessor(M, K, (void*)(B + (j+3) * K), (void*)(&(LUT_Scales[3])), (void*)QLUT3);  
        float32x4_t lut_scales_vec = vld1q_f32(LUT_Scales);  // Load all 4 LUT scales

        //printf("LUTs constructed for rows %d-%d, scale=%.2f\n", j, j+3, *LUT_Scales);  
        
        // Parallelize over row blocks
        #pragma omp parallel for num_threads(4)
        for (int ii = 0; ii < M; ii += BM) {          
            for (int kk = 0; kk < KK; kk += BK) {
                int8x16_t vec_lut0_high[BK];
                int8x16_t vec_lut0_low[BK];
                int8x16_t vec_lut1_high[BK];
                int8x16_t vec_lut1_low[BK];
                int8x16_t vec_lut2_high[BK];
                int8x16_t vec_lut2_low[BK];
                int8x16_t vec_lut3_high[BK];
                int8x16_t vec_lut3_low[BK];
                
                // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
                for (int k = 0; k < BK; k++) {
                    vec_lut0_high[k] = vld1q_s8(QLUT0 + (kk + k) * 32);      // Load high bytes
                    vec_lut0_low[k] = vld1q_s8(QLUT0 + (kk + k) * 32 + 16);   // Load low bytes
                    vec_lut1_high[k] = vld1q_s8(QLUT1 + (kk + k) * 32);      // Load high bytes
                    vec_lut1_low[k] = vld1q_s8(QLUT1 + (kk + k) * 32 + 16);   // Load low bytes
                    vec_lut2_high[k] = vld1q_s8(QLUT2 + (kk + k) * 32);      // Load high bytes
                    vec_lut2_low[k] = vld1q_s8(QLUT2 + (kk + k) * 32 + 16);   // Load low bytes
                    vec_lut3_high[k] = vld1q_s8(QLUT3 + (kk + k) * 32);      // Load high bytes
                    vec_lut3_low[k] = vld1q_s8(QLUT3 + (kk + k) * 32 + 16);   // Load low bytes
                }
                
#pragma unroll
                for (int i = ii; i < ii + BM; i += 16) {
                    int16x8_t vec_c0[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
                    int16x8_t vec_c1[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
                    int16x8_t vec_c2[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
                    int16x8_t vec_c3[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
#pragma unroll
                    for (int k = kk; k < kk + BK; k++) {
                        uint8x16_t vec_a = vld1q_u8(A + k * M + i);

                        int8x16_t vec_c0_h = vqtbl1q_s8(vec_lut0_high[k - kk], vec_a);
                        int8x16_t vec_c0_l = vqtbl1q_s8(vec_lut0_low[k - kk], vec_a);
                        int8x16_t vec_c1_h = vqtbl1q_s8(vec_lut1_high[k - kk], vec_a);
                        int8x16_t vec_c1_l = vqtbl1q_s8(vec_lut1_low[k - kk], vec_a);
                        int8x16_t vec_c2_h = vqtbl1q_s8(vec_lut2_high[k - kk], vec_a);
                        int8x16_t vec_c2_l = vqtbl1q_s8(vec_lut2_low[k - kk], vec_a);
                        int8x16_t vec_c3_h = vqtbl1q_s8(vec_lut3_high[k - kk], vec_a);
                        int8x16_t vec_c3_l = vqtbl1q_s8(vec_lut3_low[k - kk], vec_a);

                        int16x8_t out00, out01;
                        int16x8_t out10, out11;
                        int16x8_t out20, out21;
                        int16x8_t out30, out31;
                        reconstruct_int16_pair(vec_c0_h, vec_c0_l, out00, out01);
                        vec_c0[0] = vaddq_s16(vec_c0[0], out00);
                        vec_c0[1] = vaddq_s16(vec_c0[1], out01);
                        reconstruct_int16_pair(vec_c1_h, vec_c1_l, out10, out11);
                        vec_c1[0] = vaddq_s16(vec_c1[0], out10);
                        vec_c1[1] = vaddq_s16(vec_c1[1], out11);
                        reconstruct_int16_pair(vec_c2_h, vec_c2_l, out20, out21);
                        vec_c2[0] = vaddq_s16(vec_c2[0], out20);
                        vec_c2[1] = vaddq_s16(vec_c2[1], out21);
                        reconstruct_int16_pair(vec_c3_h, vec_c3_l, out30, out31);
                        vec_c3[0] = vaddq_s16(vec_c3[0], out30);
                        vec_c3[1] = vaddq_s16(vec_c3[1], out31);
                    }

                    int32x4_t vec_out_lo[4];
                    int32x4_t vec_out_hi[4];
                    int32x4_t vec_out_lo2[4];
                    int32x4_t vec_out_hi2[4];
                    interleave_vec_c_block(vget_low_s16(vec_c0[0]), vget_low_s16(vec_c1[0]), vget_low_s16(vec_c2[0]), vget_low_s16(vec_c3[0]), vec_out_lo);
                    interleave_vec_c_block(vget_high_s16(vec_c0[0]), vget_high_s16(vec_c1[0]), vget_high_s16(vec_c2[0]), vget_high_s16(vec_c3[0]), vec_out_hi);
                    interleave_vec_c_block(vget_low_s16(vec_c0[1]), vget_low_s16(vec_c1[1]), vget_low_s16(vec_c2[1]), vget_low_s16(vec_c3[1]), vec_out_lo2);
                    interleave_vec_c_block(vget_high_s16(vec_c0[1]), vget_high_s16(vec_c1[1]), vget_high_s16(vec_c2[1]), vget_high_s16(vec_c3[1]), vec_out_hi2);
                    
                    //convert int32 to float32 and scale
                    float32_t* pC = &C[(i+0)*N + j];
                    
                    float32x4_t vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo[0]);
                    float32x4_t scaled_lo_0 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo_0));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo[1]);
                    float32x4_t scaled_lo_1 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo_1));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo[2]);
                    float32x4_t scaled_lo_2 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo_2));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo[3]);
                    float32x4_t scaled_lo_3 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo_3));
                    pC += N;
                    
                    float32x4_t vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi[0]);
                    float32x4_t scaled_hi_0 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi_0));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi[1]);
                    float32x4_t scaled_hi_1 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi_1));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi[2]);
                    float32x4_t scaled_hi_2 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi_2));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi[3]);
                    float32x4_t scaled_hi_3 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi_3));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo2[0]);
                    float32x4_t scaled_lo2_0 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo2_0));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo2[1]);
                    float32x4_t scaled_lo2_1 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo2_1));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo2[2]);
                    float32x4_t scaled_lo2_2 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo2_2));
                    pC += N;
                    
                    vec_out_lo_f32 = vcvtq_f32_s32(vec_out_lo2[3]);
                    float32x4_t scaled_lo2_3 = vmulq_f32(vec_out_lo_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_lo2_3));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi2[0]);
                    float32x4_t scaled_hi2_0 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi2_0));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi2[1]);
                    float32x4_t scaled_hi2_1 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi2_1));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi2[2]);
                    float32x4_t scaled_hi2_2 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi2_2));
                    pC += N;
                    
                    vec_out_hi_f32 = vcvtq_f32_s32(vec_out_hi2[3]);
                    float32x4_t scaled_hi2_3 = vmulq_f32(vec_out_hi_f32, lut_scales_vec);
                    vst1q_f32(pC, vaddq_f32(vld1q_f32(pC), scaled_hi2_3));
                }
            }
        }
    }

    aligned_free(QLUT0);
    aligned_free(QLUT1);
    aligned_free(QLUT2);
    aligned_free(QLUT3);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

/* A(K/4 x M), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_packed(uint8_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    int KK = K / 4;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 2.0f;

    for (int j = 0; j < N; j++) {
        ggml_preprocessor(M, K, (void*)(B + j * K), (void*)LUT_Scales, (void*)QLUT);                  
        //printf("LUT constructed for row %d, scale=%.2f\n", j, *LUT_Scales);    
        
        // Parallelize over row blocks
        #pragma omp parallel for num_threads(4)
        for (int ii = 0; ii < M; ii += BM) {          
            for (int kk = 0; kk < KK; kk += BK) {
                int8x16_t vec_lut_high[BK];
                int8x16_t vec_lut_low[BK];
                
                // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
                for (int k = 0; k < BK; k++) {
                    vec_lut_high[k] = vld1q_s8(QLUT + (kk + k) * 32);      // Load high bytes
                    vec_lut_low[k] = vld1q_s8(QLUT + (kk + k) * 32 + 16);   // Load low bytes
                }
                
#pragma unroll
                for (int i = ii; i < ii + BM; i += 16) {
                    int16x8_t vec_c[4] = {vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0)};
#pragma unroll
                    for (int k = kk; k < kk + BK; k++) {
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
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

/* A(K/2 x M), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_micro_kernel(uint8_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    int ne00 = M;
    int ne01 = K;
    int ne10 = K;
    int ne11 = N;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;

    #pragma omp parallel num_threads(4)
    {    
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();
    
        for (int j = 0; j < ne11; j++) {
            if (ith == 0) {
                ggml_preprocessor(ne00, ne10, B + (j * ne10), LUT_Scales, QLUT);
            }
#pragma omp barrier

            const int range_per_thread_ii = ne00 / nth;
            for (int ii = ith * range_per_thread_ii; ii < (ith + 1) * range_per_thread_ii; ii += BM) {          
                ggml_qgemm_lut( ne00, ne11, ne10, ii, j, A, 
                                QLUT, 
                                Scales, 
                                LUT_Scales, 
                                C);
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

void compare_matrices(float32_t* C_simd, float32_t* C_, int M, int N, float32_t threshold, const char* label) {
    float32_t max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        float32_t error = fabs((float32_t)C_simd[i] - C_[i]);
        if (error > max_error) {
            max_error = error;
        }
        if ((error / (fabs((float32_t)C_simd[i]) + 1e-6)) > threshold) {  // Threshold for significant error
            error_count++;
            if (error_count <= 10) {  // Print first 10 errors
                printf("  Mismatch at [%d]: kernel=%.1f, ref=%.1f, error=%.1f\n", 
                       i, C_simd[i], C_[i], error);
            }
        }
    }
    printf("%s: max_error=%.1f, mismatches=%d/%d\n", label, max_error, error_count, M * N);
}

int main() {    
    printf("Allocating matrices...\n");
    
    // Allocate matrices
    float32_t* B = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    float32_t* B_T = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    uint8_t* A = (uint8_t*)aligned_malloc(M * K / 2 * sizeof(uint8_t));
    //uint8_t* A_T = (uint8_t*)aligned_malloc(M * K / 2 * sizeof(uint8_t));
    uint8_t* A_packed = (uint8_t*)aligned_malloc(M * K / 4 * sizeof(uint8_t));
    uint8_t* A_packed_T = (uint8_t*)aligned_malloc(M * K / 4 * sizeof(uint8_t));
    int8_t* A_ = (int8_t*)aligned_malloc(M * K * sizeof(int8_t));
    //int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    float32_t* C_ = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    float32_t* C_simd = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    
    // Allocate reference output matrix C_
    memset(C_, 0, M * N * sizeof(float32_t));
    //memset(C, 0, M * N * sizeof(int32_t));
    memset(C_simd, 0, M * N * sizeof(float32_t));

    // Initialize with random values
    printf("Initializing test matrices...\n");
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> distr(-32.0f, 32.0f);    
    for (int i = 0; i < N * K; i++) {
        B[i] = distr(gen);
    }

    transpose_matrix(B, B_T, N, K);

    for (int i = 0; i < M * K / 2; i++) {
        A[i] = rand() % 9;

        switch(A[i]) {
            case 0: A_[i * 2 + 0] = -1; A_[i * 2 + 1] = -1; break;
            case 1: A_[i * 2 + 0] = -1; A_[i * 2 + 1] = 0;  break;
            case 2: A_[i * 2 + 0] = -1; A_[i * 2 + 1] = 1; break;
            case 3: A_[i * 2 + 0] = 0; A_[i * 2 + 1] = -1; break;
            case 4: A_[i * 2 + 0] = 0; A_[i * 2 + 1] = 0; break;
            case 5: A_[i * 2 + 0] = 0; A_[i * 2 + 1] = 1; break;
            case 6: A_[i * 2 + 0] = 1; A_[i * 2 + 1] = -1; break;
            case 7: A_[i * 2 + 0] = 1; A_[i * 2 + 1] = 0; break;
            case 8: A_[i * 2 + 0] = 1; A_[i * 2 + 1] = 1; break;
        }
        
        if(i > 0 && i % 2 == 0) {
            A_packed[i / 2 -1] = (uint8_t)((A[i-2] << 4) | (A[i - 1] & 0x0F));
        }
    }
    A_packed[M * K / 4 -1] = (uint8_t)((A[M * K / 2 - 2] << 4) | (A[M * K / 2 - 1] & 0x0F));

    // Transpose A for SIMD version
    int KK = K / 2;
    //transpose_matrix(A, A_T, KK, M);
    transpose_matrix(A_packed, A_packed_T, KK /2, M);
  
    printf("Running LUT construction and inference...\n");
    printf("Matrix dimensions:  A(2560x2560), B(2560x640), C(2560x160)\n");

    // Step 0: Compute reference result using normal matmul (A_ @ B.T -> C_)
    printf("\nStep 0: Computing reference matmul with A_ and B...\n");
    // C_[m,n] = sum_k A_[n,k] * B[m,k]
    auto naive_start = std::chrono::high_resolution_clock::now();
    matmul_naive(A_, B, C_, M, N, K);
    auto naive_end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start);
    
    printf("Reference matmul complete. Time: %lld ms\n", naive_duration.count());
    
    // Step 1: Run qGEMM with LUT
    /*printf("\nStep 1: Running qGEMM_LUT (32x64 kernel)\n");
    auto lut_start = std::chrono::high_resolution_clock::now();
    matmul_lut_naive2(A, B_T, C, M, N, K);
    auto lut_end = std::chrono::high_resolution_clock::now();
    auto lut_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_end - lut_start);
    
    printf("Matmul_naive2 complete. Time: %lld ms\n", lut_duration.count());*/
    
    // Step 2: Run qGEMM with LUT + SIMD (100 runs for averaging)
    //printf("\nStep 2: Running qGEMM_LUT SIMD (100 iterations for average)\n");
    const int num_iterations = 10;
    /*long long total_simd_time = 0;
    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C_simd, 0, M * N * sizeof(int32_t));
        auto lut_simd_start = std::chrono::high_resolution_clock::now();
        matmul_lut_simd(A_T, B_T, C_simd, M, N, K);
        auto lut_simd_end = std::chrono::high_resolution_clock::now();
        auto lut_simd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_simd_end - lut_simd_start);
        total_simd_time += lut_simd_duration.count();
    }
    long long avg_simd_time = total_simd_time / num_iterations;
    printf("Matmul_simd complete. Average time over %d runs: %lld ms\n", num_iterations, avg_simd_time);

    printf("\nComparing kernel output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-2, "Matmul_simd comparison");

           
    // Step 3: Run qGEMM with LUT + SIMD (100 runs for averaging)
    printf("\nStep 3: Running qGEMM_LUT SIMD2 (100 iterations for average)\n");
    long long total_simd_time2 = 0;
    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C_simd, 0, M * N * sizeof(float32_t));
        auto lut_simd_start2 = std::chrono::high_resolution_clock::now();
        matmul_lut_simd2(A_T, B_T, C_simd, M, N, K);
        auto lut_simd_end2 = std::chrono::high_resolution_clock::now();
        auto lut_simd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_simd_end2 - lut_simd_start2);
        total_simd_time2 += lut_simd_duration.count();
    }
    long long avg_simd_time2 = total_simd_time2 / num_iterations;
    printf("Matmul_simd2 complete. Average time over %d runs: %lld ms\n", num_iterations, avg_simd_time2);

    printf("\nComparing kernel output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-2, "Matmul_simd2 comparison");*/

    // Step 3: Run qGEMM with micro kernel (100 runs for averaging)
    printf("\nStep 3: Running qGEMM_LUT microkernel (100 iterations for average)\n");
    long long total_microkernel_time = 0;
    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C_simd, 0, M * N * sizeof(float32_t));
        auto microkernel_start = std::chrono::high_resolution_clock::now();           
        matmul_lut_packed(A_packed_T, B_T, C_simd, M, N, K);
        auto microkernel_end = std::chrono::high_resolution_clock::now();
        auto microkernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(microkernel_end - microkernel_start);
        total_microkernel_time += microkernel_duration.count();
    }
    long long avg_microkernel_time = total_microkernel_time / num_iterations;
    printf("Matmul_microkernel complete. Average time over %d runs: %lld ms\n", num_iterations, avg_microkernel_time);
    printf("\nComparing kernel output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-2, "Matmul_microkernel comparison");

    // Print performance comparison
    //double speedup_naive2 = (double)naive_duration.count() / (double)lut_duration.count();
    //double speedup_simd = (double)naive_duration.count() / (double)avg_simd_time;
    //double speedup_simd2 = (double)naive_duration.count() / (double)avg_simd_time2;
    double speedup_microkernel = (double)naive_duration.count() / (double)avg_microkernel_time;
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("matmul naive:   %lld ms\n", naive_duration.count());
    //printf("LUT matmul SIMD (avg):   %lld ms\n", avg_simd_time);
    //printf("Speedup (naive / SIMD): %.2fx\n\n", speedup_simd);
    //printf("LUT matmul SIMD2 (avg):   %lld ms\n", avg_simd_time2);
    //printf("Speedup (naive / SIMD2): %.2fx\n\n", speedup_simd2);
    printf("LUT matmul microkernel (avg):   %lld ms\n", avg_microkernel_time);
    printf("Speedup (naive / microkernel): %.2fx\n\n", speedup_microkernel);
    
    // Cleanup
    aligned_free(C_);
    aligned_free(B);
    aligned_free(A);
    //aligned_free(A_T);
    aligned_free(A_packed_T);
    aligned_free(A_);
    //aligned_free(C);
    aligned_free(A_packed);
    aligned_free(B_T);
    aligned_free(C_simd);
    
    printf("\nTest complete.\n");
    return 0;
}

