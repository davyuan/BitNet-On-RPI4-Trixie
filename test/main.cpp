#define GGML_BITNET_ARM_TL1 ON
#include <cstdlib>
#include <cstring>
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

    for(int i=0; i< N; i++){       
        // Step 2: Run qGEMM with LUT
        printf("\nStep 2: Running qGEMM_LUT (640x2560 kernel) iteration %d...\n", i);
        qgemm_lut_640_2560(A + i * K / 4, QLUT, Scales, LUT_Scales, C+i*M);
        
        // Check guards after each iteration
        if ((i + 1) % 100 == 0 || i == N - 1) {
            check_all_guards();
        }
    }
    
    printf("Matmul complete.\n");
    check_all_guards();
    
    printf("Sample output values (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
    
    // Cleanup
    aligned_free(LUT_Scales);
    aligned_free(Scales);
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

