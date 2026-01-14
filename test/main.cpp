#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include "./bitnet-lut-kernels.h"

int main() {
    const int M = 1;           // Activation rows (B rows)
    const int K = 2560;        // Shared dimension
    const int N = 640;         // Weight rows (A rows) = output size
    
    // Allocate matrices
    // B: activation matrix (M x K) = (1 x 2560)
    float32_t* B = (float32_t*)aligned_malloc(M *K * sizeof(float32_t));
    
    // A: weight matrix (N x K) = (640 x 2560)
    uint8_t* A = (uint8_t*)aligned_malloc(N * K / 4);  // 2-bit quantized
    int8_t* A_ = (int8_t*)aligned_malloc(N * K);  
    
    // C: output matrix (M x N) = (1 x 640)
    int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    memset(C, 0, M * N * sizeof(int32_t));
    
    // LUT tables: every weight pair (2 weights) has 16 bytes of LUT values
    int8_t* QLUT = (int8_t*)aligned_malloc(K / 2 * 16);  // LUT for K weights
    
    // Scales
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    
    // Initialize with random values
    printf("Initializing test matrices...\n");
    for (int i = 0; i < M * K; i++) {
        B[i] = (float32_t)(rand() % 256);
    }

    for (int i = 0; i < N * K / 4; i++) {
        uint8_t high = rand() % 9;
        uint8_t low = rand() % 9;
        A[i] = (high << 4) | low;

        switch(high) {
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

        switch(low) {
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
    
    // Set scales to reasonable values
    *LUT_Scales = 1.0f;
    *Scales = 1.0f;
    
    printf("Running LUT construction and inference...\n");
    printf("Matrix dimensions: B(1x2560), A(640x2560), C(1x640)\n");
    
    // Step 1: Build LUT from weight matrix A (first row for testing)
    printf("\nStep 1: Building LUT table...\n");
    
    lut_ctor<K>(QLUT, B, LUT_Scales);
    printf("LUT construction complete. LUT_Scales = %f\n", *LUT_Scales);

    for(int i=0; i< N; i++){       
        // Step 2: Run qGEMM with LUT
        printf("\nStep 2: Running qGEMM_LUT (640x2560 kernel)...\n");
        qgemm_lut_640_2560(A + i * K / 4, QLUT, Scales, LUT_Scales, C+i*M);
    }
    
    printf("Matmul complete.\n");
    printf("Sample output values (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
    
    // Cleanup
    aligned_free(LUT_Scales);
    aligned_free(Scales);
    printf("freeing B...\n");  
    aligned_free(B);
    printf("freeing A, A_, C, and QLUT...\n");  
    aligned_free(A);
    aligned_free(A_);
    aligned_free(C);
    aligned_free(QLUT);
    
    printf("\nTest complete.\n");
    return 0;
}

