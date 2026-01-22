# CPU Backend Graph Computation Analysis
## llama.cpp GGML Framework - Thread Parallelization and MatMul Task Distribution

**Location:** `/3rdparty/llama.cpp/ggml/src/`  
**Key Files:** 
- `ggml-backend.cpp` - Backend interface and CPU backend implementation
- `ggml.c` - Core computation engine with thread pool and parallelization

---

## Overview

The llama.cpp GGML framework implements CPU-based tensor computation with multi-threaded parallelization. The computation graph is executed through a sophisticated task distribution system that splits work across threads.

---

## 1. CPU Backend Entry Point

### File: [ggml-backend.cpp](ggml-backend.cpp#L942-L961)

**Function:** `ggml_backend_cpu_graph_compute()`

```cpp
static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    // Create a computation plan based on the graph and thread count
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);

    // Allocate work buffer if needed
    if (cpu_ctx->work_size < cplan.work_size) {
        delete[] cpu_ctx->work_data;
        cpu_ctx->work_data = new uint8_t[cplan.work_size];
        // ... error handling ...
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    // Set abort callback
    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    // Execute the computation plan
    return ggml_graph_compute(cgraph, &cplan);
}
```

**Key Points:**
- Wraps the graph computation with proper resource allocation
- Thread count is taken from CPU context
- Work buffer is dynamically allocated based on computation plan
- Optional abort callback support for early termination

---

## 2. Computation Plan Creation

### File: [ggml.c](ggml.c#L20271-L20450)

**Function:** `ggml_graph_plan()`

This function analyzes the computation graph and determines:
1. Number of tasks for each operation
2. Work buffer size requirements

```c
struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    // ... initialization ...
    
    int max_tasks = 1;
    size_t work_size = 0;

    // For each node in the graph, determine parallelization strategy
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // Key call: determine how many tasks to create for this operation
        const int n_tasks = ggml_get_n_tasks(node, n_threads);
        max_tasks = MAX(max_tasks, n_tasks);

        // Estimate memory needed for this operation
        size_t cur = 0;
        // ... operation-specific memory calculation ...
        work_size = MAX(work_size, cur);
    }

    // ... create cplan structure ...
}
```

### Operation-Specific Task Assignment

#### For Matrix Multiply (MUL_MAT):

```c
case GGML_OP_MUL_MAT:
{
    const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
    // BitNet-optimized implementation
    if (ggml_bitnet_can_mul_mat(node->src[0], node->src[1], node)) {
        cur = ggml_bitnet_mul_mat_get_wsize(node->src[0], node->src[1], node);
    } else
#endif
    // Standard implementation
    if (node->src[1]->type != vec_dot_type) {
        if (vec_dot_type == GGML_TYPE_I8_S) {
            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1])) 
                + node->src[1]->ne[1] * sizeof(float) 
                + node->src[1]->ne[1] * sizeof(int32_t);
        } else {
            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
        }
    }
} break;
```

---

## 3. Task Determination Function

### File: [ggml.c](ggml.c#L19790-L20000)

**Function:** `ggml_get_n_tasks()` - **CRITICAL FOR UNDERSTANDING PARALLELIZATION**

This function determines how many parallel tasks should be created for each operation:

```c
static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        n_tasks = 1;  // No parallelization for empty tensors
        return n_tasks;
    }

    switch (node->op) {
        // Operations that CAN be parallelized across all threads
        case GGML_OP_MUL_MAT:           // *** MATRIX MULTIPLY ***
        case GGML_OP_MUL_MAT_ID:        // *** INDEXED MATRIX MULTIPLY ***
        case GGML_OP_OUT_PROD:
        case GGML_OP_ADD:
        case GGML_OP_ROPE:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        {
            n_tasks = n_threads;        // *** FULL THREAD PARALLELIZATION ***
        } break;

        // Operations that run on limited threads
        case GGML_OP_SOFT_MAX:
        {
            n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
        } break;

        // Operations that must be single-threaded
        case GGML_OP_SUM:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
        {
            n_tasks = 1;               // *** NO PARALLELIZATION ***
        } break;

        // ... more cases ...
    }
    
    return n_tasks;
}
```

**KEY INSIGHT:** For MUL_MAT, `n_tasks = n_threads` means the work is split into as many chunks as there are threads.

---

## 4. Main Graph Computation Function

### File: [ggml.c](ggml.c#L20735-L20800)

**Function:** `ggml_graph_compute()` - **THREAD ORCHESTRATION**

This is where the actual parallel computation happens:

```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);

    int n_threads = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        // Create temporary thread pool if not provided
        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
        disposable_threadpool = true;
    } else {
        // Reset threadpool state for new computation
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = false;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    // *** OpenMP parallel region ***
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // Update thread count from actual OpenMP threads
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
            }

            // Each thread executes graph computation
            ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    // *** Manual thread pool (no OpenMP) ***
    if (n_threads > threadpool->n_threads_max) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick off all worker threads with new work
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // Main thread also participates in computation
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // Ensure no thread affinity left on main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}
```

**KEY PARALLELIZATION STRATEGIES:**
1. **OpenMP Path** (`GGML_USE_OPENMP`): Uses `#pragma omp parallel` for implicit thread management
2. **Custom Threadpool Path**: Manual thread pool with explicit work distribution via `ggml_graph_compute_kickoff()`

---

## 5. Per-Thread Computation Function

### File: [ggml.c](ggml.c#L20460-L20490)

**Function:** `ggml_graph_compute_thread()` - **WHERE ACTUAL WORK HAPPENS**

```c
static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    // Set NUMA affinity for this thread
    set_numa_thread_affinity(state->ith);

    // Prepare computation parameters for this thread
    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,           // Thread index
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),  // Total threads
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    // MAIN LOOP: Process each node in the computation graph
    for (int node_n = 0; node_n < cgraph->n_nodes && !tp->abort; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        // *** Dispatch computation based on operation type ***
        // This calls operation-specific functions like ggml_compute_forward_mul_mat()
        ggml_compute_forward(&params, node);

        // Check for abort signal from main thread
        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            tp->abort = true;
            tp->ec    = GGML_STATUS_ABORTED;
        }

        // *** BARRIER: Synchronize all threads before next node ***
        // This ensures all threads finish their work on current node
        // before any thread moves to the next node
        ggml_barrier(state->threadpool);
    }

    return 0;
}
```

**CRITICAL PATTERN:**
1. Each thread receives the full computation graph
2. Each thread processes all nodes sequentially
3. At the barrier after each node, threads wait for each other
4. Within each node's computation, tasks are subdivided across threads

---

## 6. Operation Dispatch - ggml_compute_forward()

### File: [ggml.c](ggml.c#L17812-L17950)

**Function:** `ggml_compute_forward()` - **OPERATION-SPECIFIC DISPATCH**

```c
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        // ... many operations ...
        
        case GGML_OP_MUL_MAT:
            {
                // *** Routes to matrix multiplication implementation ***
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
            
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
            
        // ... more operations ...
    }
}
```

---

## 7. MatMul Implementation - The Real Work

### File: [ggml.c](ggml.c#L12585-L12710)

**Function:** `ggml_compute_forward_mul_mat()` - **WHERE MATMUL TASKS ARE SPLIT**

This is where the actual thread splitting happens for matrix multiplication:

```c
static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];  // Weight matrix
    const struct ggml_tensor * src1 = dst->src[1];  // Activation matrix

    const int ith = params->ith;   // *** Current thread index ***
    const int nth = params->nth;   // *** Total number of threads ***

    // *** BITNET-SPECIFIC IMPLEMENTATION (ARM TL1) ***
#if defined(GGML_BITNET_ARM_TL1)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
        const int bits = ggml_bitnet_get_type_bits(type);
        
        char * wdata = params->wdata;
        struct bitnet_tensor_extra * wt = src0->extra;
        
        // ... preprocessing and data setup ...
        
        // TASK DISTRIBUTION LOOP
        for(int j = 0; j < ne11; j++) {  // Loop over output columns (batches)
            
            // Single-threaded preprocessing (only thread 0)
            if (ith == 0) {
                ggml_bitnet_transform_tensor(src0);
                ggml_preprocessor(ne00, ne10, act_input + (j * ne10), lut_scales, qlut);
            }

            // *** BARRIER: Wait for preprocessing to complete ***
            ggml_barrier(params->threadpool);

            // *** THREAD WORK DISTRIBUTION ***
            // Each thread computes a different range of output rows
            const int range_per_thread_ii = ne00 / nth;  // Divide output rows by thread count
            for (int ii = ith * range_per_thread_ii;     // Start: thread_idx * chunk_size
                       ii < (ith + 1) * range_per_thread_ii;  // End: (thread_idx + 1) * chunk_size
                       ii += BM) {  // BM is block size for cache optimization
                
                // *** ACTUAL COMPUTATION: Quantized GEMM with LUT ***
                ggml_qgemm_lut(
                    ne00, ne11, ne10, ii, j,
                    ((uint8_t *)(wt->qweights)), 
                    qlut, 
                    wt->scales, 
                    // ... more parameters ...
                );
            }
        }
    }
#endif

    // *** BITNET-SPECIFIC IMPLEMENTATION (X86 TL2) ***
#if defined(GGML_BITNET_X86_TL2)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
        // Similar pattern to TL1, with optimized kernels for x86
        // ... (specialized x86 SIMD optimizations)
    }
#endif

    // *** STANDARD IMPLEMENTATION (NON-BITNET) ***
    // Uses ggml_compute_forward_mul_mat_one_chunk() for generic matmul
}
```

### Block-Tiled MatMul Computation

### File: [ggml.c](ggml.c#L12407-L12550)

**Function:** `ggml_compute_forward_mul_mat_one_chunk()` - **THREAD WORK SUBDIVISION**

For non-BitNet operations, uses block tiling:

```c
static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,   // *** Thread's start row ***
    const int64_t ir0_end,     // *** Thread's end row ***
    const int64_t ir1_start,
    const int64_t ir1_end) {

    // ... setup code ...

    // Threads with no work simply yield
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    // Block-tiling parameters for cache optimization
    const int64_t blck_0 = 16;  // Row block size
    const int64_t blck_1 = 16;  // Column block size

    // *** NESTED LOOPS: Each thread works on its assigned ranges ***
    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                // ... compute output location ...

                // *** VECTOR DOT COMPUTATION ***
                // Each thread computes its assigned block of output
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 
                            src0_row + ir0 * nb01, 
                            src1_col);
                }

                // Store results to output matrix
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

---

## 8. Thread Synchronization

### Barrier Function

The `ggml_barrier()` function synchronizes threads between graph nodes:

```c
// Each thread waits at the barrier until all threads have finished
// their current operation, then proceeds to the next operation
ggml_barrier(state->threadpool);
```

This ensures:
1. **Data Consistency**: All results from one operation are visible before the next
2. **Sequential Ordering**: Graph nodes execute in dependency order
3. **Load Balancing**: Threads wait for slowest thread before continuing

---

## 9. Threadpool Management (Non-OpenMP)

### File: [ggml.c](ggml.c#L20608-L20720)

**Function:** `ggml_graph_compute_kickoff()` - **THREAD POOL ACTIVATION**

When not using OpenMP, this function wakes up worker threads:

```c
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Signal all threads that new work is available
    // Threads transition from polling/sleeping to active computation
}
```

---

## 10. Work Distribution Pattern Summary

### For Matrix Multiply with N threads:

```
Input:  Weight matrix (M x K), Activation matrix (K x N)
Output: Result matrix (M x N)

Thread 0: Computes output rows 0 to M/N
Thread 1: Computes output rows M/N to 2*M/N
Thread 2: Computes output rows 2*M/N to 3*M/N
...
Thread N-1: Computes output rows (N-1)*M/N to M

Each thread:
├── Wait for preprocessing (barrier)
├── Compute its row chunk independently
├── Store results to output matrix
└── Wait at barrier for next operation
```

---

## 11. BitNet-Specific Optimizations

### Location: [ggml.c](ggml.c#L12653-L12705) (ARM TL1)

Key BitNet-specific features:

1. **Quantized GEMM with LUT**: `ggml_qgemm_lut()`
   - Uses lookup tables for quantized weights
   - Much faster than standard fp32 operations

2. **Tensor Transformation**: `ggml_bitnet_transform_tensor()`
   - Prepares quantized weights for efficient computation

3. **Preprocessing**: `ggml_preprocessor()`
   - Converts activations to integer form for quantized operations

4. **Thread-Safe Work Division**:
   - Thread 0 does preprocessing (synchronized)
   - All threads compute independently on their row ranges
   - Minimal synchronization overhead

---

## 12. Performance Characteristics

### Thread Scaling

```
Single thread:   Max latency
Multiple threads: Latency / min(N_threads, work_items)

For MatMul (M x K @ K x N):
- Max benefit when M is divisible by N_threads
- Each thread computes M/N_threads rows independently
- Communication overhead limited to barriers
```

### Cache Optimization

- **Block Tiling** (16x16): Improves cache locality
- **NUMA Awareness**: `set_numa_thread_affinity()` pins threads to NUMA nodes
- **Contiguous Data**: Minimizes memory access patterns

---

## Summary

The llama.cpp CPU backend implements sophisticated multi-threaded tensor computation through:

1. **Graph Planning** - Determines parallelization strategy for each operation
2. **Thread Pool** - Manages worker threads (OpenMP or custom)
3. **Work Distribution** - Divides matrix operations across threads
4. **Synchronization** - Barriers ensure correct data dependencies
5. **Optimization** - Block tiling, NUMA awareness, and BitNet-specific kernels

For **matrix multiplication specifically**:
- Work divided by **output rows** across threads
- Each thread computes its **row chunk independently**
- **BitNet** uses quantized GEMM with lookup tables for fast computation
- Thread synchronization only at operation boundaries (barriers)
