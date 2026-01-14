#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "./bitnet-lut-kernels.h"

#define CANARY_VALUE 0xDEADBEEF
#define CANARY_SIZE 64  // bytes

struct AllocGuard {
    const char* name;
    uint32_t* ptr;
    size_t size;
    uint32_t* canary_before;
    uint32_t* canary_after;
};

AllocGuard guards[10];
int guard_count = 0;

AllocGuard* create_guard(const char* name, size_t size) {
    // Allocate: canary_before | data | canary_after
    size_t total = CANARY_SIZE + size + CANARY_SIZE;
    uint32_t* ptr = (uint32_t*)aligned_malloc(total);
    
    AllocGuard* g = &guards[guard_count++];
    g->name = name;
    g->canary_before = ptr;
    g->ptr = (uint32_t*)((uint8_t*)ptr + CANARY_SIZE);
    g->canary_after = (uint32_t*)((uint8_t*)g->ptr + size);
    g->size = size;
    
    // Initialize canaries
    for (int i = 0; i < CANARY_SIZE / 4; i++) {
        g->canary_before[i] = CANARY_VALUE;
        g->canary_after[i] = CANARY_VALUE;
    }
    
    printf("Allocated %s: %zu bytes (with guards)\n", name, size);
    return g;
}

bool check_guard(AllocGuard* g) {
    bool ok = true;
    
    // Check before canary
    for (int i = 0; i < CANARY_SIZE / 4; i++) {
        if (g->canary_before[i] != CANARY_VALUE) {
            printf("ERROR: %s BEFORE canary corrupted at offset %d\n", g->name, i * 4);
            ok = false;
            break;
        }
    }
    
    // Check after canary
    for (int i = 0; i < CANARY_SIZE / 4; i++) {
        if (g->canary_after[i] != CANARY_VALUE) {
            printf("ERROR: %s AFTER canary corrupted at offset %d\n", g->name, i * 4);
            ok = false;
            break;
        }
    }
    
    if (ok) {
        printf("✓ %s canaries OK\n", g->name);
    }
    return ok;
}

void check_all_guards() {
    printf("\n=== Checking all memory guards ===\n");
    bool all_ok = true;
    for (int i = 0; i < guard_count; i++) {
        if (!check_guard(&guards[i])) {
            all_ok = false;
        }
    }
    if (all_ok) {
        printf("All guards OK!\n");
    } else {
        printf("MEMORY OVERFLOW DETECTED!\n");
    }
    printf("===================================\n\n");
}

int main() {
    const int M = 160;           // Activation rows (B rows)
    const int K = 2560;        // Shared dimension
    const int N = 640;         // Weight rows (A rows) = output size
    
    printf("Allocating matrices with overflow guards...\n");
    
    // Allocate matrices with guards
    AllocGuard* g_B = create_guard("B", M * K * sizeof(float32_t));
    AllocGuard* g_A = create_guard("A", N * K / 4);
    AllocGuard* g_A_ = create_guard("A_", N * K);
    AllocGuard* g_C = create_guard("C", M * N * sizeof(int32_t));
    AllocGuard* g_QLUT = create_guard("QLUT", K * 16);
    
    // Cast to actual types
    float32_t* B = (float32_t*)g_B->ptr;
    uint8_t* A = (uint8_t*)g_A->ptr;
    int8_t* A_ = (int8_t*)g_A_->ptr;
    int32_t* C = (int32_t*)g_C->ptr;
    int8_t* QLUT = (int8_t*)g_QLUT->ptr;
    
    float32_t* LUT_Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    float32_t* Scales = (float32_t*)aligned_malloc(sizeof(float32_t));
    
    // Allocate reference output matrix C_
    float32_t* C_ = (float32_t*)aligned_malloc(M * N * sizeof(float32_t));
    memset(C_, 0, M * N * sizeof(float32_t));
    
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
    check_all_guards();
    
    // Debug: Print first 8 B value pairs and corresponding LUT values
    printf("\n=== DEBUG: First 8 B pairs and corresponding LUT ===\n");
    for (int idx = 0; idx < 8; idx++) {
        printf("\nB pair %d: B[%d]=%.1f, B[%d]=%.1f\n", 
               idx, idx*2, B[idx*2], idx*2+1, B[idx*2+1]);
        
        // Print corresponding LUT values (256 bytes per index in the 9-LUT table)
        printf("  LUT[%d] (256 bytes total, showing first 32):\n", idx);
        int8_t* lut_ptr = QLUT + idx * 256;
        for (int i = 0; i < 32; i++) {
            if (i % 16 == 0) printf("    [%2d]: ", i);
            printf("%3d ", lut_ptr[i]);
            if ((i + 1) % 16 == 0) printf("\n");
        }
    }
    printf("=== END DEBUG ===\n\n");

    // Step 2: Run qGEMM with LUT
    printf("\nStep 2: Running qGEMM_LUT (640x2560 kernel)\n");
    for(int i=0; i< N; i++){       
        qgemm_lut_640_2560(A + i * K / 4, QLUT, Scales, LUT_Scales, C+i*M);
        
        // Check guards after each iteration
        if ((i + 1) % 100 == 0 || i == N - 1) {
            check_all_guards();
        }
    }
    
    printf("Matmul complete.\n");
    check_all_guards();
    
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
    aligned_free(LUT_Scales);
    aligned_free(Scales);
    aligned_free(C_);
    printf("freeing B...\n");  
    aligned_free((void*)g_B->canary_before);
    printf("freeing A, A_, C, and QLUT...\n");  
    aligned_free((void*)g_A->canary_before);
    aligned_free((void*)g_A_->canary_before);
    aligned_free((void*)g_C->canary_before);
    aligned_free((void*)g_QLUT->canary_before);
    
    printf("\nTest complete.\n");
    return 0;
}

