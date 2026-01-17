#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "./bitnet-lut-kernels.h"

#define TILE_K 32
#define TILE_N 16
#define TILE_M 4
#define TILE_SIZE 16

const int BM = 160;
const int BY = 256;
const int bm = 32;
const int by = (256/(bm));
const int M = 640;           // Activation rows (B rows)
const int K = 2560;        // Shared dimension
const int N = 160;         // Weight rows (A rows) = output size

// Repack matrix A according to the tl1 layout pattern
// BM, BY, bm, by are the tiling parameters
// Input: weight_in of shape (M, K//2) flattened
// Output: weight_out of shape (M*K//64, 16) flattened
void process_tl1(const uint8_t* input_weight, uint8_t* output_weight, 
                     int M, int K, int BM, int BY, int bm, int by) {
    // The Python code packs two 4-bit weights into one byte at the end.
    // The input 'input_weight' is assumed to be M * (K/2) bytes.
    
    int out_idx = 0;

    // We follow the hierarchical tiling: BM (Large M block) -> BY (Large K block)
    for (int i_major = 0; i_major < M; i_major += BM) {
        for (int j_major = 0; j_major < K; j_major += BY) {
            
            // bm (Sub-block M) -> by (Sub-block K)
            for (int i_minor = 0; i_minor < BM; i_minor += bm) {
                for (int j_minor = 0; j_minor < BY; j_minor += by) {
                    
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
            B_T[j * N + i] = B[i * K + j];
        }
    }
}

void matmul_lut(int8_t* A, float32_t* B, int32_t* C, int M, int N, int K) {
    int KK = K / 2;
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    *Scales = 1.0f;
    *LUT_Scales = 1.0f;

    // Partition rows among 4 cores
    #pragma omp parallel for num_threads(4) 
    for (int ii = 0; ii < M; ii += TILE_SIZE) {          
        for (int jj = 0; jj < N; jj += TILE_SIZE) {      
            for (int kk = 0; kk < KK; kk += TILE_SIZE) {                
                for (int i = ii; i < std::min(ii + TILE_SIZE, M); i++) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); j++) {                        
                        lut_ctor<2560>(QLUT, (float32_t*)(B + j* K), LUT_Scales);    
                        int32_t local_sum = 0; 
                        
                        for (int k = kk; k < std::min(kk + TILE_SIZE, KK); k++) {
                            int8_t high_byte = QLUT[k * 32 + A[i*KK + k]];
                            uint8_t low_byte = (uint8_t)QLUT[k * 32 + 16 + A[i*KK + k]];
                            int16_t combined = ((int16_t)high_byte << 8) | low_byte;
                            local_sum += (int32_t)combined;
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

int main() {    
    printf("Allocating matrices with overflow guards...\n");
    
    // Allocate matrices
    float32_t* B = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    float32_t* B_T = (float32_t*)aligned_malloc(N * K * sizeof(float32_t));
    int8_t* A = (int8_t*)aligned_malloc(M * K / 2 * sizeof(int8_t));
    int8_t* A_ = (int8_t*)aligned_malloc(M * K * sizeof(int8_t));
    uint8_t* A_packed = (uint8_t*)aligned_malloc(M * K / 4 * sizeof(uint8_t));
    int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    float32_t* C_ = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    
    // Allocate reference output matrix C_
    memset(C_, 0, M * N * sizeof(float32_t));
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
    
    // Debug: Print sample elements from A matrix for sanity check
    printf("\n=== DEBUG: Sample A matrix elements ===\n");
    printf("A matrix (first 16 bytes, hex representation):\n");
    for (int i = 0; i < 128; i++) {
        uint8_t high = (A[i] >> 4) & 0xF;
        uint8_t low = A[i] & 0xF;
        printf("A[%2d] = 0x%02x (high=%d, low=%d)", i, A[i], high, low);
        
        // Show corresponding A_ values
        printf(" -> A_[%3d..%3d] = [%2d %2d %2d %2d]\n", 
               i*4, i*4+3, 
               A_[i*4+0], A_[i*4+1], A_[i*4+2], A_[i*4+3]);
    }
    printf("=== END DEBUG ===\n\n");
    
    printf("Running LUT construction and inference...\n");
    printf("Matrix dimensions:  A(640x2560), B(160x2560), C(640x160)\n");
    
    // Debug: Print first 8 B value pairs and corresponding LUT values
    /*printf("\n=== DEBUG: First 8 B pairs and corresponding LUT ===\n");
    for (int idx = 0; idx < 8; idx++) {
        printf("\nB pair %d: B[%d]=%.1f, B[%d]=%.1f\n", 
               idx, idx*2, B[idx*2], idx*2+1, B[idx*2+1]);
        
        // Print corresponding LUT values (32 bytes per index)
        // First 16 bytes are high bytes, second 16 are low bytes
        printf("  LUT[%d] - high bytes:\n", idx);
        int8_t* lut_ptr = QLUT + idx * 32;
        for (int i = 0; i < 16; i++) {
            printf("%3d ", lut_ptr[i]);
        }
        printf("\n");
        
        printf("  LUT[%d] - low bytes:\n", idx);
        for (int i = 0; i < 16; i++) {
            printf("%3d ", lut_ptr[16 + i]);
        }
        printf("\n");
        
        printf("  LUT[%d] - reconstructed int16 values:\n", idx);
        for (int i = 0; i < 16; i++) {
            int16_t val = ((int16_t)lut_ptr[i] << 8) | (lut_ptr[16 + i] & 0xFF);
            printf("%6d ", val);
        }
        printf("\n");
    }
    printf("=== END DEBUG ===\n\n");*/

    // Step 2: Run qGEMM with LUT
    printf("\nStep 2: Running qGEMM_LUT (640x2560 kernel)\n");
    matmul_lut(A, B_T, C, M, N, K);
    
    printf("Matmul complete.\n");
    
    // Step 3: Compute reference result using normal matmul (A_ @ B.T -> C_)
    printf("\nStep 3: Computing reference matmul with A_ and B...\n");
    // C_[m,n] = sum_k A_[n,k] * B[m,k]
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float32_t sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += (float32_t)A_[n * K + k] * B[m * K + k];
            }
            C_[m * N + n] = sum;
        }
    }
    printf("Reference matmul complete.\n");
    
    // Step 4: Compare results
    printf("\nStep 4: Comparing kernel output (C) with reference (C_)...\n");
    float32_t max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; i++) {
        float32_t error = fabs((float32_t)C[i] - C_[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (error > 1e-3) {  // Threshold for significant error
            error_count++;
            if (error_count <= 10) {  // Print first 10 errors
                printf("  Mismatch at [%d]: kernel=%d, ref=%.1f, error=%.1f\n", 
                       i, C[i], C_[i], error);
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
    
    printf("\nTest complete.\n");
    return 0;
}

