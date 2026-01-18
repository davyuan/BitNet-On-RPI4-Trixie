#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "./bitnet-lut-kernels.h"

#define TILE_K 32
#define TILE_N 16
#define TILE_M 4
#define TILE_SIZE 2
#define K_DIM 2560

const int BM = 32;
const int BK = 32;
const int bm = 32;
const int by = (256/(bm));
const int M =640;           // Activation rows (B rows)
const int K = 2560;        // Shared dimension
const int N = 160;         // Weight rows (A rows) = output size

// Repack matrix A according to the tl1 layout pattern
// BM, BY, bm, by are the tiling parameters
// Input: weight_in of shape (M, K//2) flattened
// Output: weight_out of shape (M*K//64, 16) flattened
void process_tl1(const uint8_t* input_weight, uint8_t* output_weight, 
                     int M, int K, int BM, int BK, int bm, int bk) {
    // The Python code packs two 4-bit weights into one byte at the end.
    // The input 'input_weight' is assumed to be M * (K/2) bytes.
    
    int out_idx = 0;

    // We follow the hierarchical tiling: BM (Large M block) -> BY (Large K block)
    for (int i_major = 0; i_major < M; i_major += BM) {
        for (int j_major = 0; j_major < K; j_major += BK) {
            
            // bm (Sub-block M) -> by (Sub-block K)
            for (int i_minor = 0; i_minor < BM; i_minor += bm) {
                for (int j_minor = 0; j_minor < BK; j_minor += bk) {
                    
                    // Hardware Atoms: 16 rows (bm_inner) x 4 columns (by_inner)
                    for (int i_atom = 0; i_atom < bm; i_atom += 16) {
                        for (int j_atom = 0; j_atom < by; j_atom += 4) {
                            
                            // Inside the 16x4 Atom
                            for (int r = 0; r < 16; ++r) {
                                // Python logic: weight = weight_0 << 4 + weight_1
                                // weight_0 comes from index 0, weight_1 from index 1 of the last dim
                                // In the K dimension, index 0 and 1 are 2-bits apart in the packed byte.
                                
                                int row = i_major + i_minor + i_atom + r;
                                int col_pair = (j_major + j_minor + j_atom) / 2;

                                // Load the byte containing the 4-bit weights
                                // In NumPy: weight = weight.reshape(..., 4 // 2, 16)
                                uint8_t val = input_weight[row * (K / 2) + col_pair];
                                uint8_t val_next = input_weight[row * (K / 2) + col_pair + 1];

                                // Extract and shift as per the Python bit-packing
                                // weight_0 = weight[:, :, 0] << 4
                                // weight_1 = weight[:, :, 1]
                                uint8_t w0 = (val & 0x0F) << 4;   // Assuming low nibble is weight_0
                                uint8_t w1 = (val_next & 0x0F);    // Assuming low nibble is weight_1
                                
                                output_weight[out_idx++] = w0 | w1;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Transpose matrix B from (K x N) to B_T (N x K)
void transpose_matrix(float32_t* B, float32_t* B_T, int N, int K) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B_T[j * K + i] = B[i * N + j];
        }
    }
}

void matmul_naive(int8_t* A, float32_t* B, int32_t* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
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
                        lut_ctor<K_DIM>(QLUT, (float32_t*)(B + j* K), LUT_Scales);    
                        
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

    // Partition rows among 4 cores
    #pragma omp parallel for num_threads(4) 
    for (int j = 0; j < N; j++) {                        
        lut_ctor<K_DIM>(QLUT, (float32_t*)(B + j* K), LUT_Scales);    
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

/* A(MxK/2), B(NxK)
   QLUT(K*16), QLUT is contructed for each row of B. each K has 32 bytes (first 16 high bytes and then 16 low bytes)
        each K represents 2 activations in B. 
   C(MxN)
   This version doesn't use SIMD optimizations either, but focus on one LUT table at once to avoid
   overhead of reconstructing LUTs in the same tile. 
*/
void matmul_lut_simd(int8_t* A, float32_t* B, int32_t* C, int M, int N, int K) {
    int KK = K / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 1.0f;

    // Partition rows among 4 cores
    #pragma omp parallel for num_threads(4) 
    for (int j = 0; j < N; j++) {                        
        lut_ctor<K_DIM>(QLUT, (float32_t*)(B + j* K), LUT_Scales);    
        for (int ii = 0; ii < M; ii += BM) {          
            for (int kk = 0; kk < KK; kk += BK) {                
                /*for (int i = ii; i < ii + BM; i++) {
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
                }*/

                // Load LUT high and low byte tables separately
                int8x16_t vec_lut_high[BK];
                int8x16_t vec_lut_low[BK];
                
                // LUT layout per index: [16 high_bytes] [16 low_bytes] = 32 bytes
#pragma unroll
                for (int k = 0; k < BK; k++) {
                    vec_lut_high[k] = vld1q_s8(QLUT + (kk + k) * 32);      // Load high bytes
                    vec_lut_low[k] = vld1q_s8(QLUT + (kk + k) * 32 + 16);   // Load low bytes (offset by all high bytes)
                }
#pragma unroll
                for (int i = ii; i < ii + BM; i ++) {
                    int16x8_t vec_c[4] = {vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0), vdupq_n_s16(0)};

#pragma unroll
                     for (int k = kk; k < kk + BK; k+= 32) {
                        int8x16_t vec_a0 = vld1q_s8(A + i * KK + k * 32);
                        
                        // Lookup on high and low tables separately
                        int8x16_t vec_c0_h = vqtbl1q_s8(vec_lut_high[k - kk], vec_a0);
                        int8x16_t vec_c0_l = vqtbl1q_s8(vec_lut_low[k - kk], vec_a0);
                        
                        // Reconstruct int16 from high/low bytes: (high << 8) | low
                        int16x8_t v0h_lo_16 = vshlq_n_s16(vmovl_s8(vget_low_s8(vec_c0_h)), 8);
                        int16x8_t v0h_hi_16 = vshlq_n_s16(vmovl_s8(vget_high_s8(vec_c0_h)), 8);  
                        int16x8_t v0l_lo_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_low_s8(vec_c0_l))));
                        int16x8_t v0l_hi_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_high_s8(vec_c0_l))));

                        // combine
                        int16x8_t out0 = vorrq_s16(v0h_lo_16, v0l_lo_16);
                        int16x8_t out1 = vorrq_s16(v0h_hi_16, v0l_hi_16);
                        
                        vec_c[0] += out0;
                        vec_c[1] += out1;

                        int8x16_t vec_a1 = vld1q_s8(A + i * KK + k * 32 + 16);
                        
                        // Lookup on high and low tables separately
                        int8x16_t vec_c1_h = vqtbl1q_s8(vec_lut_high[k - kk + 16], vec_a1);
                        int8x16_t vec_c1_l = vqtbl1q_s8(vec_lut_low[k - kk + 16], vec_a1);
                        
                        // Reconstruct int16 from high/low bytes: (high << 8) | low
                        int16x8_t v1h_lo_16 = vshlq_n_s16(vmovl_s8(vget_low_s8(vec_c1_h)), 8);
                        int16x8_t v1h_hi_16 = vshlq_n_s16(vmovl_s8(vget_high_s8(vec_c1_h)), 8);  
                        int16x8_t v1l_lo_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_low_s8(vec_c1_l))));
                        int16x8_t v1l_hi_16 = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_s8(vget_high_s8(vec_c1_l))));

                        // combine
                        int16x8_t out2 = vorrq_s16(v1h_lo_16, v1l_lo_16);
                        int16x8_t out3 = vorrq_s16(v1h_hi_16, v1l_hi_16);
                        
                        vec_c[2] += out2;
                        vec_c[3] += out3;
                    }

                    int16_t sum = vaddvq_s16(vec_c[0]) + vaddvq_s16(vec_c[1]) + vaddvq_s16(vec_c[2]) + vaddvq_s16(vec_c[3]);
                    C[i*N + j] += sum;

                    /*int32x4_t vec_c0_low = vmovl_s16(vget_low_s16(vec_c[0]));
                    int32x4_t vec_c0_high = vmovl_high_s16(vec_c[0]);
                    vst1q_s32(C + i * N + j + 0, vld1q_s32(C + i * N + j + 0) + vec_c0_low);
                    vst1q_s32(C + i * N + j + 4, vld1q_s32(C + i * N + j + 4) + vec_c0_high);
                    int32x4_t vec_c1_low = vmovl_s16(vget_low_s16(vec_c[1]));
                    int32x4_t vec_c1_high = vmovl_high_s16(vec_c[1]);
                    vst1q_s32(C + i * N + j + 8, vld1q_s32(C + i * N + j + 8) + vec_c1_low);
                    vst1q_s32(C + i * N + j + 12, vld1q_s32(C + i * N + j + 12) + vec_c1_high);
                    int32x4_t vec_c2_low = vmovl_s16(vget_low_s16(vec_c[2]));
                    int32x4_t vec_c2_high = vmovl_high_s16(vec_c[2]);
                    vst1q_s32(C + i * N + j + 16, vld1q_s32(C + i * N + j + 16) + vec_c2_low);
                    vst1q_s32(C + i * N + j + 20, vld1q_s32(C + i * N + j + 20) + vec_c2_high);
                    int32x4_t vec_c3_low = vmovl_s16(vget_low_s16(vec_c[3]));
                    int32x4_t vec_c3_high = vmovl_high_s16(vec_c[3]);
                    vst1q_s32(C + i * N + j + 24, vld1q_s32(C + i * N + j + 24) + vec_c3_low);
                    vst1q_s32(C + i * N + j + 28, vld1q_s32(C + i * N + j + 28) + vec_c3_high);*/
                }
            }
        }
    }

    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
}

int main() {    
    printf("Allocating matrices...\n");
    
    // Allocate matrices
    float32_t* B = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    float32_t* B_T = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    int8_t* A = (int8_t*)aligned_malloc(M * K / 2 * sizeof(int8_t));
    int8_t* A_ = (int8_t*)aligned_malloc(M * K * sizeof(int8_t));
    uint8_t* A_packed = (uint8_t*)aligned_malloc(M * K / 4 * sizeof(uint8_t));
    int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    int32_t* C_ = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    int32_t* C_simd = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    
    // Allocate reference output matrix C_
    memset(C_, 0, M * N * sizeof(int32_t));
    memset(C, 0, M * N * sizeof(int32_t));

    // Initialize with random values
    printf("Initializing test matrices...\n");
    for (int i = 0; i < N * K; i++) {
        B[i] = (float32_t)(rand() % 256);
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
    }

    // Repack A into tl1 layout
    printf("Repacking matrix A into tl1 layout...\n");
    //process_tl1(A, A_packed, M, K, BM, BY, bm, by);
    
    printf("Running LUT construction and inference...\n");
    printf("Matrix dimensions:  A(640x2560), B(2560x160), C(640x160)\n");

    // Step 2: Run qGEMM with LUT
    printf("\nStep 2: Running qGEMM_LUT (32x64 kernel)\n");
    auto lut_start = std::chrono::high_resolution_clock::now();
    matmul_lut_naive2(A, B_T, C, M, N, K);
    auto lut_end = std::chrono::high_resolution_clock::now();
    auto lut_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_end - lut_start);
    
    printf("Matmul_naive2 complete. Time: %lld ms\n", lut_duration.count());
    
    // Step 3: Run qGEMM with LUT + SIMD
    printf("\nStep 3: Running qGEMM_LUT (32x64 kernel)\n");
    auto lut_simd_start = std::chrono::high_resolution_clock::now();
    matmul_lut_simd(A, B_T, C_simd, M, N, K);
    auto lut_simd_end = std::chrono::high_resolution_clock::now();
    auto lut_simd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(lut_simd_end - lut_simd_start);
    
    printf("Matmul_simd complete. Time: %lld ms\n", lut_simd_duration.count());

    // Step 4: Compute reference result using normal matmul (A_ @ B.T -> C_)
    printf("\nStep 4: Computing reference matmul with A_ and B...\n");
    // C_[m,n] = sum_k A_[n,k] * B[m,k]
    auto naive_start = std::chrono::high_resolution_clock::now();
    matmul_naive(A_, B, (int32_t*)C_, M, N, K);
    auto naive_end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start);
    
    printf("Reference matmul complete. Time: %lld ms\n", naive_duration.count());
    
    // Print performance comparison
    double speedup_naive2 = (double)naive_duration.count() / (double)lut_duration.count();
    double speedup_simd = (double)naive_duration.count() / (double)lut_simd_duration.count();
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("Naive matmul: %lld ms\n", naive_duration.count());
    printf("LUT matmul native2:   %lld ms\n", lut_duration.count());
    printf("LUT matmul SIMD:   %lld ms\n", lut_simd_duration.count());
    printf("Speedup naive2: %.2fx\n", speedup_naive2);
    printf("Speedup SIMD: %.2fx\n\n", speedup_simd);
    
    // Step 4: Compare results
    printf("\nStep 4: Comparing kernel output (C) with reference (C_)...\n");
    float32_t max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        float32_t error = fabs((float32_t)C_simd[i] - C_[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > 1e-3) {  // Threshold for significant error
            error_count++;
            if (error_count <= 10) {  // Print first 10 errors
                printf("  Mismatch at [%d]: kernel=%d, ref=%.1f, error=%.1f\n", 
                       i, C_simd[i], C_[i], error);
            }
        }
    }
    printf("Comparison complete: max_error=%.1f, mismatches=%d/%d\n", 
           max_error, error_count, M * N);
    
    // Cleanup
    aligned_free(C_);
    aligned_free(B);
    aligned_free(A);
    aligned_free(A_);
    aligned_free(C);
    aligned_free(A_packed);
    aligned_free(B_T);
    aligned_free(C_simd);
    
    printf("\nTest complete.\n");
    return 0;
}

