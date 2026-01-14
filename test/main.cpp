#define GGML_BITNET_ARM_TL1 ON

#include "bitnet-lut-kernels.h"
#include <arm_neon.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdlib.h>

int main() {
    const int M = 1;           // Activation rows (B rows)
    const int K = 2560;        // Shared dimension
    const int N = 640;         // Weight rows (A rows) = output size
    
    // Allocate matrices
    // B: activation matrix (M x K) = (1 x 2560)
    float* B = (float*)aligned_malloc(M *K * sizeof(float));
    
    // A: weight matrix (N x K) = (640 x 2560)
    uint8_t* A = (uint8_t*)aligned_malloc(N * K / 4);  // 2-bit quantized
    int8_t* A_ = (int8_t*)aligned_malloc(N * K);  
    
    // C: output matrix (M x N) = (1 x 640)
    int32_t* C = (int32_t*)aligned_malloc(M * N * sizeof(int32_t));
    memset(C, 0, M * N * sizeof(int32_t));
    
    // LUT tables
    int8_t* QLUT = (int8_t*)aligned_malloc(K * 16 * sizeof(int8_t));  // LUT for K weights
    
    // Scales
    float* LUT_Scales = (float*)aligned_malloc(sizeof(float));
    float* Scales = (float*)aligned_malloc(sizeof(float));
    
    // Initialize with random values
    printf("Initializing test matrices...\n");
    for (int i = 0; i < M * K; i++) {
        B[i] = (float)(rand() % 256);
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
    
    for(int i=0; i< N; i++){
        // Step 1: Build LUT from weight matrix A (first row for testing)
        printf("\nStep 1: Building LUT table...\n");
        
        //per_tensor_quant(K, LUT_Scales, B_float);
        lut_ctor<K>(QLUT, B+ i*K, LUT_Scales);
        printf("LUT construction complete. LUT_Scales = %f\n", *LUT_Scales);
        
        // Step 2: Run qGEMM with LUT
        printf("\nStep 2: Running qGEMM_LUT (640x2560 kernel)...\n");
        qgemm_lut_640_2560(A, QLUT, Scales, LUT_Scales, C+i*M);
    }
    
    printf("Matmul complete.\n");
    printf("Sample output values (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
    
    // Cleanup
    aligned_free(B);
    aligned_free(A);
    aligned_free(C);
    aligned_free(QLUT);
    aligned_free(LUT_Scales);
    aligned_free(Scales);
    
    printf("\nTest complete.\n");
    return 0;
}

