#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <omp.h>

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

const int M = 640;           // Activation rows (B rows)
const int K = 2560;        // Shared dimension
const int N = 160;         // Weight rows (A rows) = output size
const int TILE_SIZE = 16;  // Tile size for blocking

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

    memset(C, 0, M * N * sizeof(int32_t));
    memset(C_, 0, M * N * sizeof(int32_t));

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
    
    // Calculate speedup
    double speedup = (double)naive_duration.count() / (double)tiled_duration.count();
    printf("Speedup: %.2fx\n\n", speedup);

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_[i]) {
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
}