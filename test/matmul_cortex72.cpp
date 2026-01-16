#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <arm_neon.h>

#define TILE_K 32
#define TILE_N 16
#define TILE_M 4 // Register block height
#define TILE_SIZE 16  // Tile size for blocking
const int M  = 640;           // Activation rows (B rows)
const int K  = 2560;        // Shared dimension
const int N  = 160;         // Weight rows (A rows) = output size

static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void matmul_tiled_simd(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < M; i += TILE_M) {
        for (int j = 0; j < N; j += TILE_N) {            
            // 1. Initialize 4x4 registers (total 16 accumulators)
            // Each register holds four 32-bit results
            // Row 0 Accumulators (16 elements total)
            int32x4_t c00 = vdupq_n_s32(0);
            int32x4_t c01 = vdupq_n_s32(0);
            int32x4_t c02 = vdupq_n_s32(0);
            int32x4_t c03 = vdupq_n_s32(0);

            // Row 1 Accumulators
            int32x4_t c10 = vdupq_n_s32(0);
            int32x4_t c11 = vdupq_n_s32(0);
            int32x4_t c12 = vdupq_n_s32(0);
            int32x4_t c13 = vdupq_n_s32(0);

            // Row 2 Accumulators
            int32x4_t c20 = vdupq_n_s32(0);
            int32x4_t c21 = vdupq_n_s32(0);
            int32x4_t c22 = vdupq_n_s32(0);
            int32x4_t c23 = vdupq_n_s32(0);

            // Row 3 Accumulators
            int32x4_t c30 = vdupq_n_s32(0);
            int32x4_t c31 = vdupq_n_s32(0);
            int32x4_t c32 = vdupq_n_s32(0);
            int32x4_t c33 = vdupq_n_s32(0);

            for (int k = 0; k < K; k++) {
                // 2. Load 16 elements from B once
                int8x16_t vb = vld1q_s8(&B[k * N + j]);

                // 3. Process 4 rows of A for this k
                // We broadcast A[i][k], A[i+1][k]... and multiply by the same vb
                #define COMPUTE_ROW(r, acc0, acc1, acc2, acc3) { \
                    int8x16_t va = vdupq_n_s8(A[(i+r) * K + k]); \
                    int16x8_t prod_l = vmull_s8(vget_low_s8(va), vget_low_s8(vb)); \
                    int16x8_t prod_h = vmull_s8(vget_high_s8(va), vget_high_s8(vb)); \
                    acc0 = vaddw_s16(acc0, vget_low_s16(prod_l)); \
                    acc1 = vaddw_s16(acc1, vget_high_s16(prod_l)); \
                    acc2 = vaddw_s16(acc2, vget_low_s16(prod_h)); \
                    acc3 = vaddw_s16(acc3, vget_high_s16(prod_h)); \
                }

                COMPUTE_ROW(0, c00, c01, c02, c03);
                COMPUTE_ROW(1, c10, c11, c12, c13);
                COMPUTE_ROW(2, c20, c21, c22, c23);
                COMPUTE_ROW(3, c30, c31, c32, c33);
            }

            // 4. Store the 4x16 block back to Matrix C
            vst1q_s32(&C[(i+0)*N+j+0], c00); vst1q_s32(&C[(i+0)*N+j+4], c01);
            vst1q_s32(&C[(i+0)*N+j+8], c02); vst1q_s32(&C[(i+0)*N+j+12], c03);
            vst1q_s32(&C[(i+1)*N+j+0], c10); vst1q_s32(&C[(i+1)*N+j+4], c11);
            vst1q_s32(&C[(i+1)*N+j+8], c12); vst1q_s32(&C[(i+1)*N+j+12], c13);
            vst1q_s32(&C[(i+2)*N+j+0], c20); vst1q_s32(&C[(i+2)*N+j+4], c21);
            vst1q_s32(&C[(i+2)*N+j+8], c22); vst1q_s32(&C[(i+2)*N+j+12], c23);
            vst1q_s32(&C[(i+3)*N+j+0], c30); vst1q_s32(&C[(i+3)*N+j+4], c31);
            vst1q_s32(&C[(i+3)*N+j+8], c32); vst1q_s32(&C[(i+3)*N+j+12], c33);
        }
    }
}
/*
void matmul_tiled_simd(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    // 1. Parallelize over rows (Each core gets a private slice of M)
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < M; i++) {
        for (int jj = 0; jj < N; jj += TILE_N) {
            for (int kk = 0; kk < K; kk += TILE_K) {                
                // Micro-kernel: Process 16 columns of B at once
                for (int j = jj; j < jj + TILE_N; j += 16) {                    
                    // We need four 32-bit accumulators to hold 16 results
                    int32x4_t v_sum0 = vdupq_n_s32(0);
                    int32x4_t v_sum1 = vdupq_n_s32(0);
                    int32x4_t v_sum2 = vdupq_n_s32(0);
                    int32x4_t v_sum3 = vdupq_n_s32(0);

                    // If not the first K-tile, load existing partial sums from C
                    if (kk != 0) {
                        v_sum0 = vld1q_s32(&C[i * N + j + 0]);
                        v_sum1 = vld1q_s32(&C[i * N + j + 4]);
                        v_sum2 = vld1q_s32(&C[i * N + j + 8]);
                        v_sum3 = vld1q_s32(&C[i * N + j + 12]);
                    }

                    for (int k = kk; k < kk + TILE_K; k++) {
                        // Broadcast one A element (int8) to a register
                        int8x16_t v_a = vdupq_n_s8(A[i * K + k]);
                        
                        // Load 16 elements of B
                        int8x16_t v_b = vld1q_s8(&B[k * N + j]);

                        // Multiply: int8 * int8 -> int16 (Low and High halves)
                        int16x8_t v_prod_lo = vmull_s8(vget_low_s8(v_a), vget_low_s8(v_b));
                        int16x8_t v_prod_hi = vmull_s8(vget_high_s8(v_a), vget_high_s8(v_b));

                        // Accumulate: int16 -> int32
                        v_sum0 = vaddw_s16(v_sum0, vget_low_s16(v_prod_lo));
                        v_sum1 = vaddw_s16(v_sum1, vget_high_s16(v_prod_lo));
                        v_sum2 = vaddw_s16(v_sum2, vget_low_s16(v_prod_hi));
                        v_sum3 = vaddw_s16(v_sum3, vget_high_s16(v_prod_hi));
                    }

                    // Store final 16 int32 results back to C
                    vst1q_s32(&C[i * N + j + 0], v_sum0);
                    vst1q_s32(&C[i * N + j + 4], v_sum1);
                    vst1q_s32(&C[i * N + j + 8], v_sum2);
                    vst1q_s32(&C[i * N + j + 12], v_sum3);
                }
            }
        }
    }
}*/

void matmul_naive(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
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

void matmul_int8(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    // Partition rows among 4 cores
    #pragma omp parallel for num_threads(4) 
    for (int ii = 0; ii < M; ii += TILE_SIZE) {          
        for (int jj = 0; jj < N; jj += TILE_SIZE) {      
            for (int kk = 0; kk < K; kk += TILE_SIZE) {                
                // Micro-kernel: This part would be replaced by NEON intrinsics 
                // for actual production speed.
                for (int i = ii; i < std::min(ii + TILE_SIZE, M); i++) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); j++) {                        
                        // Local accumulator in a CPU register
                        int32_t local_sum = 0; 
                        
                        for (int k = kk; k < std::min(kk + TILE_SIZE, K); k++) {
                            // int8 * int8 -> widened to int32
                            local_sum += (int32_t)A[i*K + k] * (int32_t)B[k*N + j];
                        }

                        // Only write to memory once per tile-row
                        // If kk > 0, we are adding to a previous K-tile result
                        if (kk == 0) {
                            C[i*N + j] = local_sum;
                        } else {
                            C[i*N + j] += local_sum;
                        }
                    }
                }
            }
        }
    }
}

int main(){
    int8_t* A = (int8_t*)aligned_malloc(M * K * sizeof(int8_t));
    int8_t* A_ = (int8_t*)aligned_malloc(M * K * sizeof(int8_t));
    int8_t* B = (int8_t*)aligned_malloc(K * N * sizeof(int8_t));
    int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    int32_t* C_ = (int32_t*)aligned_malloc(M * N * sizeof(int32_t)); // Reference result
    int32_t* C_simd = (int32_t*)aligned_malloc(M * N * sizeof(int32_t)); // SIMD result

    memset(C, 0, M * N * sizeof(int32_t));
    memset(C_, 0, M * N * sizeof(int32_t));
    memset(C_simd, 0, M * N * sizeof(int32_t));

    // Initialize with random values
    printf("Initializing test matrices...\n");
    for (int i = 0; i < M * K; i++) {
        B[i] = (int8_t)(rand() % 256);
    }

    for (int i = 0; i < M * K / 4; i++) {
        uint8_t high = rand() % 9;
        uint8_t low = rand() % 9;
        A[i] = (high << 4) | low;

        //armv8 is little-endian by default
        switch(low) {
            case 0: A_[i * 4 + 0] = -1; A_[i * 4 + 1] = -1; break;
            case 1: A_[i * 4 + 0] = -1; A_[i * 4 + 1] = 0;  break;
            case 2: A_[i * 4 + 0] = -1; A_[i * 4 + 1] = 1; break;
            case 3: A_[i * 4 + 0] = 0; A_[i * 4 + 1] = -1; break;
            case 4: A_[i * 4 + 0] = 0; A_[i * 4 + 1] = 0; break;
            case 5: A_[i * 4 + 0] = 0; A_[i * 4 + 1] = 1; break;
            case 6: A_[i * 4 + 0] = 1; A_[i * 4 + 1] = -1; break;
            case 7: A_[i * 4 + 0] = 1; A_[i * 4 + 1] = 0; break;
            case 8: A_[i * 4 + 0] = 1; A_[i * 4 + 1] = 1; break;
        }

        switch(high) {
            case 0: A_[i * 4 + 2] = -1; A_[i * 4 + 3] = -1; break;
            case 1: A_[i * 4 + 2] = -1; A_[i * 4 + 3] = 0;  break;
            case 2: A_[i * 4 + 2] = -1; A_[i * 4 + 3] = 1; break;
            case 3: A_[i * 4 + 2] = 0; A_[i * 4 + 3] = -1; break;
            case 4: A_[i * 4 + 2] = 0; A_[i * 4 + 3] = 0; break;
            case 5: A_[i * 4 + 2] = 0; A_[i * 4 + 3] = 1; break;
            case 6: A_[i * 4 + 2] = 1; A_[i * 4 + 3] = -1; break;
            case 7: A_[i * 4 + 2] = 1; A_[i * 4 + 3] = 0; break;
            case 8: A_[i * 4 + 2] = 1; A_[i * 4 + 3] = 1; break;
        }        
    }

    printf("Running naive matmul for reference...\n");
    auto naive_start = std::chrono::high_resolution_clock::now();
    matmul_naive(A_, B, C_, M, N, K);
    auto naive_end = std::chrono::high_resolution_clock::now();
    auto naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start);
    printf("Naive matmul completed in %lld ms\n\n", naive_duration.count());
    
    printf("Running optimized (tiled) matmul...\n");
    auto tiled_start = std::chrono::high_resolution_clock::now();
    matmul_int8(A_, B, C, M, N, K);
    auto tiled_end = std::chrono::high_resolution_clock::now();
    auto tiled_duration = std::chrono::duration_cast<std::chrono::milliseconds>(tiled_end - tiled_start);

    printf("Tiled matmul completed in %lld ms\n\n", tiled_duration.count());
    printf("Running SIMD-optimized (tiled) matmul...\n");
    auto simd_start = std::chrono::high_resolution_clock::now();
    matmul_tiled_simd(A_, B, C_simd, M, N, K);
    auto simd_end = std::chrono::high_resolution_clock::now();
    auto simd_duration = std::chrono::duration_cast<std::chrono::milliseconds>(simd_end - simd_start);
    printf("SIMD Tiled matmul completed in %lld ms\n\n", simd_duration.count());    

    // Calculate speedup
    double speedup = (double)naive_duration.count() / (double)tiled_duration.count();
    printf("Speedup: %.2fx\n\n", speedup);
    speedup = (double)naive_duration.count() / (double)simd_duration.count();
    printf("SIMD Speedup: %.2fx\n\n", speedup);

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_simd[i]) {
            errors++;
            if (errors < 10) {
                printf("Mismatch at index %d: C=%d, C_=%d\n", i, C[i], C_[i]);
            }
        }
    }
    if (errors == 0) {
        printf("Result is correct!\n");
    } else {
        printf("Total mismatches: %d\n", errors);
    }

    // Cleanup
    aligned_free(A);
    aligned_free(A_);
    //aligned_free(B);
    //aligned_free(C);
    //aligned_free(C_);
    ///aligned_free(C_simd);
}