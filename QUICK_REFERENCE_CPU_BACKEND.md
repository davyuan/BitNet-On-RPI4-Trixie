# Quick Reference: CPU Backend Graph Compute Implementation

## File Locations

| Component | File | Key Functions |
|-----------|------|---------------|
| **Backend Interface** | `3rdparty/llama.cpp/ggml/src/ggml-backend.cpp` | `ggml_backend_cpu_graph_compute()` |
| **Core Computation** | `3rdparty/llama.cpp/ggml/src/ggml.c` | `ggml_graph_compute()` |
| **Thread Management** | `3rdparty/llama.cpp/ggml/src/ggml.c` | `ggml_graph_compute_thread()` |
| **Task Planning** | `3rdparty/llama.cpp/ggml/src/ggml.c` | `ggml_graph_plan()`, `ggml_get_n_tasks()` |
| **MatMul Compute** | `3rdparty/llama.cpp/ggml/src/ggml.c` | `ggml_compute_forward_mul_mat()` |
| **Block-Tiled MatMul** | `3rdparty/llama.cpp/ggml/src/ggml.c` | `ggml_compute_forward_mul_mat_one_chunk()` |

---

## Code Navigation Guide

### 1. Entry Point (User Code → Backend)
```cpp
// User calls (from llama.cpp or Python)
ggml_backend_graph_compute(backend, cgraph)
    ↓
// Routes to CPU backend (ggml-backend.cpp:942)
ggml_backend_cpu_graph_compute(backend, cgraph)
    ├─ Creates computation plan: ggml_graph_plan()
    ├─ Allocates work buffer
    └─ Executes: ggml_graph_compute(cgraph, &cplan)
```

### 2. Computation Planning
```c
// ggml.c:20271
ggml_graph_plan(cgraph, n_threads, threadpool)
    │
    └─ For each node in graph:
        ├─ ggml_get_n_tasks(node, n_threads)  // ggml.c:19790
        │  └─ Returns: n_tasks for this operation
        └─ Calculate work_size for operation
```

### 3. Main Graph Computation Loop
```c
// ggml.c:20735
ggml_graph_compute(cgraph, cplan)
    │
    ├─ #ifdef GGML_USE_OPENMP
    │  └─ #pragma omp parallel num_threads(n_threads)
    │     └─ Each thread: ggml_graph_compute_thread()
    │
    └─ #else
       ├─ ggml_graph_compute_kickoff() // Wake threads
       └─ Main thread: ggml_graph_compute_thread()
```

### 4. Per-Thread Graph Processing
```c
// ggml.c:20460
ggml_graph_compute_thread(worker_state)
    └─ For each node in graph:
        ├─ ggml_compute_forward(&params, node)
        │  │
        │  └─ Case GGML_OP_MUL_MAT:
        │     ├─ ggml_compute_forward_mul_mat()  // ggml.c:12585
        │     │  ├─ Check BitNet optimization
        │     │  │  └─ ggml_qgemm_lut() // Quantized kernel
        │     │  └─ Otherwise: ggml_compute_forward_mul_mat_one_chunk()
        │     │
        └─ ggml_barrier()  // Wait for all threads
```

---

## Key Functions - Detailed Breakdown

### ggml_get_n_tasks() - Task Count Determination
**File:** `ggml.c:19790`  
**Purpose:** Determines how many parallel tasks for an operation

```c
switch (node->op) {
    case GGML_OP_MUL_MAT:          // ← MATRIX MULTIPLY
    case GGML_OP_MUL_MAT_ID:       // ← INDEXED MUL_MAT
    case GGML_OP_ADD:
    case GGML_OP_ROPE:
    case GGML_OP_FLASH_ATTN_EXT:
        n_tasks = n_threads;       // ← Full parallelization
        break;
    
    case GGML_OP_SOFT_MAX:
        n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));  // Limited
        break;
    
    case GGML_OP_SUM:              // ← Reduction operations
    case GGML_OP_MEAN:
    case GGML_OP_ARGMAX:
        n_tasks = 1;               // ← Single-threaded
        break;
}
```

**Key Insight:** MUL_MAT gets `n_tasks = n_threads` → Maximum parallelization

---

### ggml_graph_compute() - Thread Orchestration
**File:** `ggml.c:20735`  
**Purpose:** Main orchestrator for parallel computation

```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    int n_threads = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)  // ← OpenMP parallelism
        {
            #pragma omp single
            {
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    // Custom threadpool path
    ggml_graph_compute_kickoff(threadpool, n_threads);  // Wake threads
    ggml_graph_compute_thread(&threadpool->workers[0]); // Main thread also works
#endif

    enum ggml_status ret = threadpool->ec;
    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }
    return ret;
}
```

**Parallelization Strategies:**
- **OpenMP:** Uses `#pragma omp parallel` for automatic thread management
- **Custom:** Uses explicit thread pool with polling/signaling

---

### ggml_graph_compute_thread() - Per-Thread Work
**File:** `ggml.c:20460`  
**Purpose:** Executed by each thread - processes all graph nodes

```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * tp = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan * cplan = tp->cplan;

    set_numa_thread_affinity(state->ith);  // ← NUMA optimization

    struct ggml_compute_params params = {
        .ith = state->ith,         // ← THIS THREAD'S INDEX
        .nth = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        .wsize = cplan->work_size,
        .wdata = cplan->work_data,
        .threadpool = tp,
    };

    // MAIN LOOP: All threads process all nodes
    for (int node_n = 0; node_n < cgraph->n_nodes && !tp->abort; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        ggml_compute_forward(&params, node);  // ← Dispatch to operation

        if (state->ith == 0 && cplan->abort_callback && 
                cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort = true;
            tp->ec = GGML_STATUS_ABORTED;
        }

        ggml_barrier(state->threadpool);  // ← SYNCHRONIZE ALL THREADS
    }

    return 0;
}
```

**Critical Pattern:**
- **All threads process all nodes** (not task queue)
- **Work split within each operation** (via `ith` and `nth`)
- **Barrier synchronizes** between nodes

---

### ggml_compute_forward_mul_mat() - MatMul Implementation
**File:** `ggml.c:12585`  
**Purpose:** Matrix multiplication - where actual parallel work happens

```c
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];  // Weight
    const struct ggml_tensor * src1 = dst->src[1];  // Activation

    const int ith = params->ith;  // ← THREAD INDEX
    const int nth = params->nth;  // ← THREAD COUNT

#if defined(GGML_BITNET_ARM_TL1)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
        // ... BitNet-specific setup ...

        for(int j = 0; j < ne11; j++) {  // For each output column

            if (ith == 0) {
                // Single-threaded preprocessing (only thread 0)
                ggml_bitnet_transform_tensor(src0);
                ggml_preprocessor(...);
            }

            ggml_barrier(params->threadpool);  // Wait for preprocessing

            // THREAD-PARALLEL COMPUTATION
            const int range_per_thread = ne00 / nth;
            for (int ii = ith * range_per_thread;           // START
                       ii < (ith + 1) * range_per_thread;   // END
                       ii += BM) {
                
                ggml_qgemm_lut(  // ← QUANTIZED GEMM with LUT
                    ne00, ne11, ne10, ii, j,
                    ((uint8_t *)(wt->qweights)),
                    qlut, wt->scales, ...
                );
            }
        }
    }
#endif

#if defined(GGML_BITNET_X86_TL2)
    // Similar BitNet x86 implementation
#endif

    // Non-BitNet path uses ggml_compute_forward_mul_mat_one_chunk()
}
```

**Thread Work Division:**
```
Thread 0: rows 0 to M/nth
Thread 1: rows M/nth to 2*M/nth
...
Thread nth-1: rows (nth-1)*M/nth to M
```

---

### ggml_compute_forward_mul_mat_one_chunk() - Block-Tiled MatMul
**File:** `ggml.c:12407`  
**Purpose:** Standard MatMul with cache-aware block tiling

```c
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,  // ← Thread's row start
    const int64_t ir0_end,    // ← Thread's row end
    const int64_t ir1_start,
    const int64_t ir1_end) {

    // Skip if no work assigned
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const int64_t blck_0 = 16;  // Row block size (cache optimization)
    const int64_t blck_1 = 16;  // Column block size

    float tmp[32];  // Temporary accumulator

    // Block-tiled loops
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {

                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                // VECTOR DOT COMPUTATION
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0],
                            src0_row + ir0 * nb01,
                            src1_col);
                }

                // Store results
                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0],
                           tmp + (cn * 16),
                           (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}
```

**Key Optimization:** Block tiling (16x16) improves L2 cache reuse by ~16x

---

## How to Find Things

### Find Matrix Multiply Implementation
1. Start: [ggml-backend.cpp:942](ggml-backend.cpp#L942) - `ggml_backend_cpu_graph_compute()`
2. → [ggml.c:20735](ggml.c#L20735) - `ggml_graph_compute()`
3. → [ggml.c:20460](ggml.c#L20460) - `ggml_graph_compute_thread()`
4. → [ggml.c:17920](ggml.c#L17920) - Dispatch to `ggml_compute_forward_mul_mat()`
5. → [ggml.c:12585](ggml.c#L12585) - Matrix multiply implementation

### Find BitNet Optimization
- [ggml.c:12653](ggml.c#L12653) - ARM TL1 BitNet (ternary/binary matmul with LUT)
- [ggml.c:12708](ggml.c#L12708) - X86 TL2 BitNet (optimized kernels)
- Look for: `ggml_bitnet_can_mul_mat()` and `ggml_qgemm_lut()`

### Find Thread Splitting Logic
- [ggml.c:19790](ggml.c#L19790) - `ggml_get_n_tasks()` - How many tasks per operation
- [ggml.c:12585-12700](ggml.c#L12585-L12700) - BitNet thread work division
- [ggml.c:12407-12550](ggml.c#L12407-L12550) - Standard block-tiled division

### Find Synchronization Points
- [ggml.c:20485](ggml.c#L20485) - Barrier in graph compute thread
- [ggml.c:12692](ggml.c#L12692) - Barrier in BitNet MatMul (preprocessing)

---

## Thread Parameters Structure

```c
struct ggml_compute_params {
    int32_t ith;           // ← Thread index (0 to nth-1)
    int32_t nth;           // ← Total number of threads
    size_t  wsize;         // ← Work buffer size
    void *  wdata;         // ← Work buffer data
    struct ggml_threadpool * threadpool;
};
```

**Usage in Operations:**
```c
const int ith = params->ith;   // My thread index
const int nth = params->nth;   // Total threads

// Divide work: each thread handles a chunk
const int chunk_size = total_items / nth;
const int start = ith * chunk_size;
const int end = (ith + 1) * chunk_size;

// Process my chunk
for (int i = start; i < end; i++) {
    // ... do work ...
}
```

---

## Performance Characteristics

### MatMul Speedup vs Threads
```
Ideal scaling: O(1/n) with n threads
Actual scaling: O(1/n) * cache_factor * memory_bandwidth_factor

Cache Optimization (Block Tiling):
├─ 16x16 blocks for L2 cache
├─ Each source row used 16 times
└─ ~16x reduction in memory bandwidth

NUMA Optimization:
├─ Pin threads to NUMA nodes
├─ Reduce cross-socket memory access
└─ ~2x speedup on NUMA systems
```

### BitNet vs Standard MatMul
```
BitNet Quantized:
├─ 1-bit or ternary weights
├─ Integer-only computation
├─ ~8-16x faster than fp32
└─ Uses specialized `ggml_qgemm_lut()` kernel

Standard (fp32/fp16):
├─ Full precision computation
├─ Generic vec_dot operations
└─ Baseline speed
```

---

## Configuration Options

### OpenMP (if available)
```c
#ifdef GGML_USE_OPENMP
    // Uses OpenMP for thread parallelism
    #pragma omp parallel num_threads(n_threads)
#else
    // Uses custom thread pool
    ggml_graph_compute_kickoff(threadpool, n_threads);
#endif
```

### BitNet Optimization
```c
#if defined(GGML_BITNET_ARM_TL1)
    // ARM ternary/binary quantized matmul
#elif defined(GGML_BITNET_X86_TL2)
    // x86 ternary/binary quantized matmul
#else
    // Standard float matmul
#endif
```

### Thread Pool
```c
struct ggml_threadpool_params {
    int n_threads;           // Number of worker threads
    ggml_threadpool_poll_t poll;
    // ... more parameters ...
};
```

---

## Summary Table

| Aspect | Implementation | Key File | Line # |
|--------|----------------|----------|--------|
| **Backend Entry** | CPU graph compute | ggml-backend.cpp | 942 |
| **Planning** | Graph analysis | ggml.c | 20271 |
| **Task Count** | Determine parallelization | ggml.c | 19790 |
| **Orchestration** | Thread coordination | ggml.c | 20735 |
| **Per-Thread** | Node processing loop | ggml.c | 20460 |
| **Operation Dispatch** | Op-specific routing | ggml.c | 17812 |
| **MatMul** | Matrix multiplication | ggml.c | 12585 |
| **BitNet MatMul** | Quantized GEMM (ARM) | ggml.c | 12653 |
| **BitNet MatMul** | Quantized GEMM (x86) | ggml.c | 12708 |
| **Block-Tiled** | Cache-optimized matmul | ggml.c | 12407 |
| **Sync** | Thread barrier | ggml.c | 20485 |
