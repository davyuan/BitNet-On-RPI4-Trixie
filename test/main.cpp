#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <arm_neon.h>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "ggml-bitnet.h"
#include "bitnet-lut-kernels.h"

#define TILE_SIZE 16
#define WM 32 // Weight block is shape (WM x BK), we use BK = 64, so each scale covers 2 K indices

const int M = 2560;           // Weight rows (A rows)
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

void matmul_naive(float32_t* A, float32_t* B, float32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// A is (M x K/2) uint8_t, B is (K x N) float32_t
void matmul_naive_weight_scale(uint8_t* A, float32_t* B, float32_t* C, float32_t* ws, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32_t sum = 0;
            for (int k = 0; k < K/2; k++) {
                uint8_t a_val = A[i*(K/2) + k];
                float32_t scale = ws[i / WM + (2*k / BK) * (M / WM)];
                float32_t b_val0 = B[(2*k)*N + j];
                float32_t b_val1 = B[(2*k + 1)*N + j];
                float32_t val = 0;
                switch(a_val){
                    case 0: val = -b_val0 - b_val1; break;
                    case 1: val = -b_val0; break;
                    case 2: val = -b_val0 + b_val1; break;
                    case 3: val = -b_val1; break;
                    case 4: val = 0; break;
                    case 5: val = b_val1; break;
                    case 6: val = b_val0 - b_val1; break;
                    case 7: val = b_val0; break;
                    case 8: val = b_val0 + b_val1; break;
                    default: assert(false); // Should not happen
                }
                sum += val * scale;
            }
            C[i*N + j] = sum;
        }
    }
}

// A is (M x K/2) uint8_t, B is (K x N) float32_t
void matmul_tiled_weight_scale(uint8_t* A, float32_t* B, float32_t* C, float32_t* ws, int M, int N, int K) {
    for(int j = 0; j < N; j++) {
        for (int ii = 0; ii < M; ii+= WM) {
            for(int kk = 0; kk < K/2; kk+=BK) {
                float32_t scale = ws[ii / WM + (2*kk / BK) * (M / WM)];
                for(int i = ii; i < ii + WM; i++) {
                    float32_t sum = 0;
                    for(int k = kk; k < kk + BK; k++) {
                        uint8_t a_val = A[i*(K/2) + k];
                        float32_t b_val0 = B[(2*k)*N + j];
                        float32_t b_val1 = B[(2*k + 1)*N + j];
                        float32_t val = 0;
                        switch(a_val){
                            case 0: val = -b_val0 - b_val1; break;
                            case 1: val = -b_val0; break;
                            case 2: val = -b_val0 + b_val1; break;
                            case 3: val = -b_val1; break;
                            case 4: val = 0; break;
                            case 5: val = b_val1; break;
                            case 6: val = b_val0 - b_val1; break;
                            case 7: val = b_val0; break;
                            case 8: val = b_val0 + b_val1; break;
                            default: assert(false); // Should not happen
                        }
                        sum += val;
                    }
                    C[i*N + j] += sum * scale;                    
                }
            }
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
void matmul_lut_simd(uint8_t* A, float32_t* B, float32_t* C, float32_t* ws, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));

    for (int j = 0; j < N; j++) {
        ggml_preprocessor(M, K, (void*)(B + j * K), (void*)LUT_Scales, (void*)QLUT);                  
       
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
                        int16x8_t out00, out01;
                        reconstruct_int16_pair(vec_c0_h, vec_c0_l, out00, out01);
                        
                        vec_c[0] = vaddq_s16(vec_c[0], out00);
                        vec_c[1] = vaddq_s16(vec_c[1], out01);
                    }

                    float32_t* pC = (float32_t*) &(C[(i+0)*N + j]);
                    const float32_t lut_scale = ((float32_t*)LUT_Scales)[0];
                    const float32_t scale = ws[0];
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

/* A(K/2 x M/2), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_packed(uint8_t* A, float32_t* B, float32_t* C, float32_t* ws, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));

    for (int j = 0; j < N; j++) {
        ggml_preprocessor(M, K, (void*)(B + j * K), (void*)LUT_Scales, (void*)QLUT);                  
        
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
                    const float32_t weight_scale = *ws;
                    int16_t tmp_vals[8];                
#pragma unroll
                    for (int block = 0; block < 4; ++block) {
                        vst1q_s16(tmp_vals, vec_c[block]);
                        for (int lane = 0; lane < 8; ++lane, pC += N) {
                            float32_t val = (tmp_vals[lane] / lut_scale) * weight_scale;
                            (*pC) += val;
                        }
                    }
                }
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
}

/* A(K/2 x M), B(N x K)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_micro_kernel(uint8_t* A, float32_t* B, float32_t* C, float32_t* ws, int M, int N, int K) {
    int ne00 = K;
    int ne01 = M;
    int ne10 = K;
    int ne11 = N;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));

    #pragma omp parallel num_threads(4)
    {    
        int ith = omp_get_thread_num();
        int nth = omp_get_num_threads();
    
        for (int j = 0; j < ne11; j++) {
            if (ith == 0) {
                ggml_preprocessor(ne01, ne00, B + (j * ne10), LUT_Scales, QLUT);
            }
#pragma omp barrier

            const int range_per_thread_ii = ne01 / nth;
            for (int ii = ith * range_per_thread_ii; ii < (ith + 1) * range_per_thread_ii; ii += BM) {          
                ggml_qgemm_lut( ne01, ne11, ne10, ii, j, A, 
                                QLUT, 
                                ws, 
                                LUT_Scales, 
                                C);
            }
#pragma omp barrier
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
}

double calculate_sqnr(const float32_t* C, const float32_t* C_hat, int M, int N) {
    double signal_power = 0.0;
    double noise_power = 0.0;
    size_t total_elements = static_cast<size_t>(M) * N;

    for (size_t i = 0; i < total_elements; ++i) {
        double ref_val = static_cast<double>(C[i]);
        double quant_val = static_cast<double>(C_hat[i]);
        double error = ref_val - quant_val;

        signal_power += ref_val * ref_val;
        noise_power += error * error;
    }

    if (noise_power == 0.0) {
        // If there is no noise, the signal is identical. 
        // Returning a high value representing infinity.
        return 999.0; 
    }

    if (signal_power == 0.0) {
        return 0.0;
    }

    return 10.0 * std::log10(signal_power / noise_power);
}

/**
 * Calculates Cosine Similarity between two matrices C and C_hat of size M x N.
 * Returns a value between -1.0 and 1.0.
 */
template<typename T>
double calculate_cosine_similarity(const float32_t* C, const T* C_hat, int M, int N) {
    double dot_product = 0.0;
    double norm_c = 0.0;
    double norm_c_hat = 0.0;
    size_t total_elements = static_cast<size_t>(M) * N;

    for (size_t i = 0; i < total_elements; ++i) {
        double val_c = static_cast<double>(C[i]);
        double val_c_hat = static_cast<double>(C_hat[i]);

        dot_product += val_c * val_c_hat;
        norm_c += val_c * val_c;
        norm_c_hat += val_c_hat * val_c_hat;
    }

    // Check for zero vectors to avoid division by zero
    if (norm_c == 0.0 || norm_c_hat == 0.0) {
        return 0.0; 
    }

    return dot_product / (std::sqrt(norm_c) * std::sqrt(norm_c_hat));
}

void compare_matrices(float32_t* C_simd, float32_t* C_, int M, int N, float32_t threshold, const char* label) {
    float32_t max_error = 0.0f;
    int max_error_idx = -1;
    int error_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    int bad_ref_count = 0;
    for (int i = 0; i < M * N; i++) {
        // Check for NaN in C_simd
        if (std::isnan(C_simd[i])) {
            nan_count++;
            if (nan_count <= 5) {  // Print first 5 NaN locations
                printf("  NaN at [%d] in C_simd\n", i);
            }
            continue;
        }
        
        // Check for Inf in C_simd
        if (std::isinf(C_simd[i])) {
            inf_count++;
            if (inf_count <= 5) {  // Print first 5 Inf locations
                printf("  Inf at [%d] in C_simd: %.1e\n", i, C_simd[i]);
            }
            continue;
        }
        
        // Check for NaN or Inf in reference
        if (std::isnan(C_[i]) || std::isinf(C_[i])) {
            bad_ref_count++;
            if (bad_ref_count <= 5) {
                printf("  Bad ref at [%d]: %.1e\n", i, C_[i]);
            }
            continue;
        }
        
        float32_t error = fabs(C_simd[i] - C_[i]);
        
        // Check if error calculation itself produced inf
        if (std::isinf(error)) {
            inf_count++;
            if (inf_count <= 5) {
                printf("  Error=Inf at [%d]: kernel=%.1e, ref=%.1e\n", i, C_simd[i], C_[i]);
            }
            continue;
        }
        
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
        if (error / (fabs(C_simd[i]) + 1e-5) > threshold) {  // Threshold for significant error
            error_count++;
            if (error_count <= 10) {  // Print first 10 errors
                printf("  Mismatch at [%d]: kernel=%.1f, ref=%.1f, error=%.1f\n", 
                       i, C_simd[i], C_[i], error);
            }
        }
    }
    printf("%s: max_error=%.1f at idx=%d (kernel=%.1e, ref=%.1e), mismatches=%d/%d, NaNs=%d, Infs=%d, BadRef=%d\n", 
           label, max_error, max_error_idx, 
           max_error_idx >= 0 ? C_simd[max_error_idx] : 0.0f,
           max_error_idx >= 0 ? C_[max_error_idx] : 0.0f,
           error_count, M * N, nan_count, inf_count, bad_ref_count);

    double sqnr = calculate_sqnr(C_, C_simd, M, N);
    double cosine_similarity = calculate_cosine_similarity(C_, C_simd, M, N);
    printf("%s: Cosine Similarity = %.6f\n", label, cosine_similarity);
    printf("%s: SQNR = %.2f dB\n", label, sqnr);
}

// BitNet 1.58 quantization: convert weights to ternary {-1, 0, 1} using absmean scaling
// Returns: pair of (quantized_weights, gamma_scale)
std::vector<int8_t> bitnet_158_quantize(const std::vector<float>& weight_array, float32_t * weight_scale, int M, int K) {
    const float32_t epsilon = 1e-7f;
    int size = weight_array.size();
    
    float sum_abs = 0.0f;
    for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
            sum_abs += std::fabs(weight_array[m * K + k]) ;
        }
    }
    float32_t gamma = sum_abs / (M * K);
    gamma = 1.097f;
    weight_scale[0] = gamma;
    
    std::vector<int8_t> quantized_w(size);    
    for(int m = 0; m < M; m++) {
        for(int k = 0; k < K; k++) {
            float32_t block_gamma = weight_scale[0];
            int idx = m * K + k;
            float normalized = weight_array[idx] / (block_gamma + epsilon);
            float rounded = std::round(normalized);
            // Clip to [-1, 1] range
            int8_t clipped = static_cast<int8_t>(
                std::max(-1.0f, std::min(1.0f, rounded))
            );
            quantized_w[idx] = clipped;
        }
    }
    
    return quantized_w;
}

// BitNet 1.58 quantization: convert weights to ternary {-1, 0, 1} using absmean scaling
// Returns: pair of (quantized_weights, gamma_scale)
std::vector<int8_t> bitnet_158_quantize_32x2(const std::vector<float>& weight_array, float32_t * weight_scale, int M, int K) {
    const float32_t epsilon = 1e-7f;
    int size = weight_array.size();
    
    for(int m = 0; m < M; m+=WM) {
        for(int k = 0; k < K/2; k++) {
            float sum_abs = 0.0f;
            for (int i = m; i < m + WM; i++) {
                sum_abs += std::fabs(weight_array[i * K + k * 2]) 
                    + std::fabs(weight_array[i * K + k * 2 + 1]);
            }
            float32_t gamma = sum_abs / (WM * 2);
            weight_scale[k * (M / WM) + (m / WM)] = gamma;
        }
    }
    
    std::vector<int8_t> quantized_w(size);    
    for(int m = 0; m < M; m+= WM) {
        for(int k = 0; k < K / 2; k++) {
            float32_t block_gamma = weight_scale[k * (M / WM) + (m / WM)];
            // Process each k within this BK block
            for (int i = m; i < m + WM; i++) {
                int idx = i * K + k * 2;
                float normalized = weight_array[idx] / (block_gamma + epsilon);
                float rounded = std::round(normalized);
                // Clip to [-1, 1] range
                int8_t clipped = static_cast<int8_t>(
                    std::max(-1.0f, std::min(1.0f, rounded))
                );
                quantized_w[idx] = clipped;

                normalized = weight_array[idx + 1] / (block_gamma + epsilon);
                rounded = std::round(normalized);
                // Clip to [-1, 1] range
                clipped = static_cast<int8_t>(
                    std::max(-1.0f, std::min(1.0f, rounded))
                );
                quantized_w[idx + 1] = clipped;
            }
        }
    }
    
    return quantized_w;
}

std::vector<int8_t> bitnet_158_quantize_32x64(const std::vector<float>& weight_array, float32_t * weight_scale, int M, int K) {
    const float32_t epsilon = 1e-7f;
    int size = weight_array.size();
    
    for(int m = 0; m < M; m+=WM) {
        for(int k = 0; k < K; k+=BK) {
            float sum_abs = 0.0f;
            for (int i = m; i < m + WM; i++) {
                for(int j = k; j < k + BK; j++) {
                    sum_abs += std::fabs(weight_array[i * K + j]) ;
                }
            }
            float32_t gamma = sum_abs / (WM * BK);
            weight_scale[(k / BK) * (M / WM) + (m / WM)] = gamma;
        }
    }
    
    std::vector<int8_t> quantized_w(size);    
    for(int m = 0; m < M; m+= WM) {
        for(int k = 0; k < K; k+=BK) {
            float32_t block_gamma = weight_scale[(k / BK) * (M / WM) + (m / WM)];
            for (int i = m; i < m + WM; i++) {
                for(int j = k; j < k + BK; j++) {
                    int idx = i * K + j;
                    float normalized = weight_array[idx] / (block_gamma + epsilon);
                    float rounded = std::round(normalized);
                    // Clip to [-1, 1] range
                    int8_t clipped = static_cast<int8_t>(
                        std::max(-1.0f, std::min(1.0f, rounded))
                    );
                    quantized_w[idx] = clipped;
                }
            }
        }
    }
    
    return quantized_w;
}

/* After packing A_packed_T will be (K/2 x M /2)
    A_ will be (M x K)
    A will be (M x K/2) 4-bit representation
    A_T will be (K/2 x M)
*/
void init_As(float32_t* A_, uint8_t* A, uint8_t* A_T, uint8_t* A_packed_T, float32_t* weight_scale, int M, int K) {
    // Load weights from binary file
    const char *weight_file = "q_proj_weights.bin";
    std::ifstream infile(weight_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open weight file: " << weight_file << "\n";
        std::cerr << "Falling back to random weights\n";
        // Fallback to random weights
        std::random_device rd; 
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> distr(-15.0f, 15.0f);    
        for (int i = 0; i < M * K; i++) {
            A_[i] = rand() % 3 -1; //distr(gen);
        }
    } else {
        // Read weights from file
        infile.read(reinterpret_cast<char*>(A_), M * K * sizeof(float32_t));
        if (!infile) {
            std::cerr << "Failed to read weights from file\n";
            return;
        }
        infile.close();
        printf("Loaded weights from %s\n", weight_file);
    }
    
    std::vector<float> A_vec(A_, A_ + M * K);
   
    // Call bitnet_158_quantize to quantize to ternary {-1, 0, 1}
    std::vector<int8_t> quantized_ternary = bitnet_158_quantize(A_vec, weight_scale, M, K);
    // Validate loaded weights: calculate cosine similarity between A_ and sign(A_)
    double cosine_sim = calculate_cosine_similarity(A_, quantized_ternary.data(), M, K);
    printf("Cosine similarity between A_ and Quantized(A_): %.6f\n", cosine_sim);
        
    // Pack ternary values into A (2 ternary values per uint8_t)
    // Map {-1, 0, 1} to indices {0-8} for 2 values: 9 combinations
    for (int i = 0; i < M * K / 2; i++) {
        int8_t val0 = quantized_ternary[i * 2];      // first value
        int8_t val1 = quantized_ternary[i * 2 + 1];  // second value
        
        A[i] = (uint8_t)(val0 * 3 + val1) + 4;  // Map to [0, 8] range
    }

    // Transpose A for SIMD version
    int KK = K / 2;
    transpose_matrix(A, A_T, KK, M);
    // Pack A into 4-bit format
    for (int i = 0; i < KK; i++) {
        for (int j = 0; j < M; j+=2) {
            uint8_t high_nibble = A_T[i * M + j] << 4;
            uint8_t low_nibble = A_T[i * M + j + 1] & 0x0F;
            A_packed_T[i * (M / 2) + j/2] = high_nibble | low_nibble;
        }
    }
}

void init_Bs(float32_t* B, float32_t* B_T, int N, int K) {
    // Initialize with random values
    printf("Initializing test matrices...\n");
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> distr(-0.08f, 0.08f);    
    for (int i = 0; i < N * K; i++) {
        B[i] = distr(gen);
    }

    transpose_matrix(B, B_T, N, K);
}

int main() {    
    printf("Allocating matrices...\n");
    
    // Allocate matrices
    float32_t* B = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    float32_t* B_T = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    uint8_t* A = (uint8_t*)aligned_malloc(M * K / 2 * sizeof(uint8_t));
    uint8_t* A_T = (uint8_t*)aligned_malloc(M * K / 2 * sizeof(uint8_t));
    uint8_t* A_packed_T = (uint8_t*)aligned_malloc(M * K / 4 * sizeof(uint8_t));
    float32_t* A_ = (float32_t*)aligned_malloc(M * K * sizeof(float32_t));
    //int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    float32_t* C_ = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    float32_t* C_simd = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    float32_t* weight_scale = (float32_t*)aligned_malloc((M / WM * K / 2) * sizeof(float32_t));
    
    // Allocate reference output matrix C_
    memset(C_, 0, M * N * sizeof(float32_t));
    //memset(C, 0, M * N * sizeof(int32_t));
    memset(C_simd, 0, M * N * sizeof(float32_t));

    init_Bs(B, B_T, N, K);
    init_As(A_, A, A_T, A_packed_T, weight_scale, M, K);
    for(int i=0; i < std::min(M / WM * K / BK, 16); i++) {
        printf("Weight scale for block %d: %.6f\n", i, weight_scale[i]);
    }

    // Debug: Print first 16 rows of A_, A_packed, and A_packed_T
    printf("\n=== DEBUG: First 16 rows of A_ (float32_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("A_[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%8.3f ", A_[i * K + j]);
        }
        printf("\n");
    }
    
    printf("\n=== DEBUG: First 16 rows of A (uint8_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("A[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%2u ", (unsigned)A[i * K/2 + j]);
        }
        printf("\n");
    }

    printf("\n=== DEBUG: First 16 rows of A_packed_T (uint8_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("A_packed_T[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%02x ", (unsigned)A_packed_T[i * M / 2 + j]);
        }
        printf("\n");
    }
    
    // Debug: Print first 16 rows of B and B_T
    printf("\n=== DEBUG: First 16 rows of B (float32_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("B[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%8.3f ", B[i * N + j]);
        }
        printf("\n");
    }

    printf("\n=== DEBUG: First 16 rows of B_T (float32_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("B_T[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%8.3f ", B_T[i * K + j]);
        }
        printf("\n");
    }

    printf("Running LUT construction and inference...\n");
    printf("Matrix dimensions:  A(2560x2560), B(2560x640), C(2560x160)\n");

    // Step 0: Compute reference result using normal matmul (A_ @ B.T -> C_)
    printf("\nStep 0: Computing reference matmul with A_ and B...\n");
    auto naive_start = std::chrono::high_resolution_clock::now();
    matmul_naive(A_, B, C_, M, N, K);
    auto naive_end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start);
    
    printf("Reference matmul complete. Time: %ld ms\n", naive_duration.count());
    printf("\nStep 2: Running tiled matmul with weight scaling, to test math stability\n");
        
    memset(C_simd, 0, M * N * sizeof(float32_t));
    /*for(int i=0; i< M/WM * K/2; i++) {
        weight_scale[i] = 1.0f;
    }*/
    //matmul_tiled_weight_scale(A, B, C_simd, weight_scale, M, N, K);
    matmul_naive_weight_scale(A, B, C_simd, weight_scale, M, N, K);
    printf("\nComparing tiled matmul with weight scaling output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-1, "Matmul_tiled_weight_scale comparison");
    
    // Debug: Print first 16 rows of C_ and C_simd
    printf("\n=== DEBUG: First 16 rows of C_ (float32_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("C_[%2df]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%8.3f ", C_[i * N + j]);
        }
        printf("\n");
    }
    printf("\n=== DEBUG: First 16 rows of C_simd (float32_t, 16 elements each) ===\n");
    for (int i = 0; i < 16; i++) {
        printf("C_simd[%2d]: ", i);
        for (int j = 0; j < 16; j++) {
            printf("%8.3f ", C_simd[i * N + j]);
        }
        printf("\n");
    }
    
    printf("\nStep 2: Running qGEMM_LUT SIMD (50 iterations for average)\n");        
    const int num_iterations = 50;
    long long total_simd_time = 0;
    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C_simd, 0, M * N * sizeof(float32_t));
        auto lut_simd_start = std::chrono::high_resolution_clock::now();
        matmul_lut_simd(A_T, B_T, C_simd, weight_scale, M, N, K);
        auto lut_simd_end = std::chrono::high_resolution_clock::now();
        auto lut_simd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_simd_end - lut_simd_start);
        total_simd_time += lut_simd_duration.count();
    }
    long long avg_simd_time = total_simd_time / num_iterations;
    printf("Matmul_lut_simd complete. Average time over %d runs: %lld ms\n", num_iterations, avg_simd_time);

    printf("\nComparing kernel output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-1, "Matmul_lut_simd comparison");

    // Step 3: Run qGEMM with micro kernel (50 runs for averaging)
    printf("\nStep 3: Running qGEMM_LUT microkernel (50 iterations for average)\n");
    
    long long total_microkernel_time = 0;
    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C_simd, 0, M * N * sizeof(float32_t));
        auto microkernel_start = std::chrono::high_resolution_clock::now();           
        matmul_lut_micro_kernel(A_packed_T, B_T, C_simd, weight_scale, M, N, K);
        auto microkernel_end = std::chrono::high_resolution_clock::now();
        auto microkernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(microkernel_end - microkernel_start);
        total_microkernel_time += microkernel_duration.count();
    }
    long long avg_microkernel_time = total_microkernel_time / num_iterations;
    printf("Matmul_microkernel complete. Average time over %d runs: %lld ms\n", num_iterations, avg_microkernel_time);
    printf("\nComparing kernel output (C) with reference (C_)...\n");
    compare_matrices(C_simd, C_, M, N, 1e-1, "Matmul_microkernel comparison");

    printf("\n");

    // Print performance comparison
    //double speedup_naive2 = (double)naive_duration.count() / (double)lut_duration.count();
    double speedup_simd = (double)naive_duration.count() / (double)avg_simd_time;
    //double speedup_simd2 = (double)naive_duration.count() / (double)avg_simd_time2;
    double speedup_microkernel = (double)naive_duration.count() / (double)avg_microkernel_time;
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("matmul naive:   %ld ms\n", naive_duration.count());
    //printf("LUT matmul SIMD (avg):   %lld ms\n", avg_simd_time);
    printf("Speedup (naive / SIMD): %.2fx\n\n", speedup_simd);
    //printf("LUT matmul SIMD2 (avg):   %lld ms\n", avg_simd_time2);
    //printf("Speedup (naive / SIMD2): %.2fx\n\n", speedup_simd2);
    printf("LUT matmul microkernel (avg):   %lld ms\n", avg_microkernel_time);
    printf("Speedup (naive / microkernel): %.2fx\n\n", speedup_microkernel);
    
    // Cleanup
    aligned_free(C_);
    aligned_free(B);
    aligned_free(A);
    aligned_free(A_T);
    aligned_free(A_packed_T);
    aligned_free(A_);
    //aligned_free(C);
    aligned_free(B_T);
    aligned_free(C_simd);
    aligned_free(weight_scale);
    
    printf("\nTest complete.\n");
    return 0;
}

