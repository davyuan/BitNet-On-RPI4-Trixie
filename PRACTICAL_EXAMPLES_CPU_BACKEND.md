# Practical Examples: CPU Backend Graph Computation

## Example 1: Simple 2D Matrix Multiply with 4 Threads

### Setup
```
Weight Matrix (src0): 16 x 8  (16 output rows, 8 input columns)
Activation Matrix (src1): 8 x 1  (8 rows, 1 column)
Output Matrix (dst): 16 x 1  (result)

Running on 4 threads
```

### ggml_get_n_tasks() Analysis
```c
// From ggml.c:19790
node->op == GGML_OP_MUL_MAT
└─ n_tasks = n_threads = 4
```

### ggml_graph_compute() Execution
```c
// From ggml.c:20735
#pragma omp parallel num_threads(4)
{
    // OpenMP creates 4 threads
    ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
}
```

### ggml_graph_compute_thread() Per-Thread Execution
```
Thread 0:
├─ params.ith = 0
├─ params.nth = 4
├─ Loop: for (int node_n = 0; node_n < cgraph->n_nodes; node_n++)
│  └─ Calls: ggml_compute_forward(&params, mul_mat_node)
│
Thread 1:
├─ params.ith = 1
├─ params.nth = 4
├─ Loop: for (int node_n = 0; node_n < cgraph->n_nodes; node_n++)
│  └─ Calls: ggml_compute_forward(&params, mul_mat_node)
│
... (same for Thread 2 and 3)
```

### ggml_compute_forward_mul_mat() - Work Division

For standard (non-BitNet) MatMul:

```c
// From ggml.c:12585

// Range per thread calculation:
const int range_per_thread = 16 / 4 = 4  // (ne00 / nth)

// Thread 0:
for (int ii = 0 * 4;           // ir0_start = 0
          ii < 1 * 4;           // ir0_end = 4
          ii += block_size) {
    // Compute output rows 0, 1, 2, 3
    vec_dot(src0_row[0], src1_col);
    vec_dot(src0_row[1], src1_col);
    vec_dot(src0_row[2], src1_col);
    vec_dot(src0_row[3], src1_col);
}

// Thread 1:
for (int ii = 1 * 4;           // ir0_start = 4
          ii < 2 * 4;           // ir0_end = 8
          ii += block_size) {
    // Compute output rows 4, 5, 6, 7
    vec_dot(src0_row[4], src1_col);
    vec_dot(src0_row[5], src1_col);
    vec_dot(src0_row[6], src1_col);
    vec_dot(src0_row[7], src1_col);
}

// Thread 2:
for (int ii = 2 * 4;           // ir0_start = 8
          ii < 3 * 4;           // ir0_end = 12
          ii += block_size) {
    // Compute output rows 8, 9, 10, 11
}

// Thread 3:
for (int ii = 3 * 4;           // ir0_start = 12
          ii < 4 * 4;           // ir0_end = 16
          ii += block_size) {
    // Compute output rows 12, 13, 14, 15
}
```

### Synchronization
```
After ggml_compute_forward() completes:
┌──────────────────────────────────────┐
│      ggml_barrier()                  │
│  (All threads wait here)             │
└──────────────────────────────────────┘
       │
       ▼
All threads proceed to next node (or exit if no more nodes)
```

---

## Example 2: BitNet MatMul with 4 Threads (ARM TL1)

### Setup
```
Weight Matrix (src0): 32 x 64  (quantized, 1-bit or ternary)
Activation Matrix (src1): 64 x 4  (4 columns → 4 parallel computations)
Output Matrix (dst): 32 x 4

Running on 4 threads (Raspberry Pi with 4 CPU cores)
```

### Step 1: Backend Setup (ggml-backend.cpp:942)
```cpp
ggml_backend_cpu_graph_compute(backend, cgraph) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, 4, threadpool);
    
    // cplan.work_size calculated for BitNet preprocessing
    // Allocate work buffer for:
    // - Quantized weights cache
    // - Preprocessed activations
    // - Thread-local buffers
    
    return ggml_graph_compute(cgraph, &cplan);
}
```

### Step 2: Graph Planning (ggml.c:20271)
```c
struct ggml_cplan ggml_graph_plan(cgraph, 4, threadpool) {
    
    for each node in graph {
        if (node->op == GGML_OP_MUL_MAT) {
            
            // ggml_get_n_tasks() → returns 4
            const int n_tasks = ggml_get_n_tasks(node, 4);
            
            // For BitNet, calculate work size
            cur = ggml_bitnet_mul_mat_get_wsize(src0, src1, dst);
            // Example: 32 * 64 * 4 * sizeof(uint8_t) for quantized cache
            
            work_size = MAX(work_size, cur);
        }
    }
    
    // Allocate total work buffer
    cplan.work_size = work_size + CACHE_LINE_SIZE * 4;  // +cache line per thread
}
```

### Step 3: Thread Orchestration (ggml.c:20735)
```c
ggml_graph_compute(cgraph, &cplan) {
    
#ifdef GGML_USE_OPENMP
    #pragma omp parallel num_threads(4)  // 4 threads
    {
        ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
    }
#endif
}
```

### Step 4: Per-Thread Processing (ggml.c:20460)
```c
ggml_graph_compute_thread(worker_state) {
    // Each of 4 threads executes this independently
    
    struct ggml_compute_params params = {
        .ith = 0 or 1 or 2 or 3,  // Thread index
        .nth = 4,                 // Total threads
        .wdata = cplan.work_data,
        .threadpool = threadpool,
    };
    
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        node = cgraph->nodes[node_n];
        
        ggml_compute_forward(&params, node);  // MUL_MAT dispatch
        
        ggml_barrier(threadpool);  // Synchronize before next node
    }
}
```

### Step 5: BitNet MatMul Computation (ggml.c:12653)
```c
ggml_compute_forward_mul_mat(&params, dst) {
#if defined(GGML_BITNET_ARM_TL1)
    if (ggml_bitnet_can_mul_mat(src0, src1, dst)) {
        
        // --- PHASE 1: Preprocessing (Single-threaded) ---
        
        for(int j = 0; j < ne11; j++) {  // 4 columns (ne11=4)
            
            if (params.ith == 0) {  // ← Only Thread 0
                // Transform quantized weights to efficient form
                ggml_bitnet_transform_tensor(src0);
                // Activation: 64-element array → integer quantization
                ggml_preprocessor(64, 4, src1 + (j*64), ...);
            }
            
            // ← All threads wait for preprocessing
            ggml_barrier(params->threadpool);
            
            // --- PHASE 2: Parallel Computation ---
            
            // Rows per thread: 32 / 4 = 8
            const int range_per_thread = 32 / 4;
            
            // Thread 0: rows 0-7
            // Thread 1: rows 8-15
            // Thread 2: rows 16-23
            // Thread 3: rows 24-31
            
            for (int ii = params.ith * range_per_thread;
                       ii < (params.ith + 1) * range_per_thread;
                       ii += 1) {  // BM block size
                
                // Quantized GEMM with LUT
                // Input: 64-element quantized activation
                // Output: 8 results (for this thread's 8 rows)
                ggml_qgemm_lut(
                    32,          // M (output rows) - for thread's portion
                    4,           // N (output columns)
                    64,          // K (inner dimension)
                    ii, j,       // Position in output matrix
                    wt->qweights,      // 1-bit quantized weights
                    qlut,              // Lookup table
                    wt->scales,        // Quantization scales
                    ...
                );
            }
        }
    }
#endif
}
```

### Timeline Visualization
```
Time ──────────────────────────────────────────────────────────────►

Thread 0: [Wait for others] ← Barrier
Thread 1: [Wait for others] ← Barrier
Thread 2: [Wait for others] ← Barrier
Thread 3: [Wait for others] ← Barrier

For column j=0:
  Thread 0:   [Transform weights, Preprocess activation] (ith==0 only)
  Thread 1:   [                  Waiting...            ]
  Thread 2:   [                  Waiting...            ]
  Thread 3:   [                  Waiting...            ]
                              ▲ Barrier point
  Thread 0:   [MatMul rows 0-7  for column 0]
  Thread 1:   [MatMul rows 8-15 for column 0]
  Thread 2:   [MatMul rows 16-23 for column 0]
  Thread 3:   [MatMul rows 24-31 for column 0]
                              ▲ Barrier point

For column j=1:
  Thread 0:   [Transform weights, Preprocess activation] (ith==0 only)
  Thread 1:   [                  Waiting...            ]
  ...
```

---

## Example 3: Mixed Operations Graph

### Graph Structure
```
Input A (16x32) ──┐
                  ├─► MatMul ──┐
Input B (32x8) ───┤            │
                  │            ├─► Add ──┐
                  │            │         │
                  │   Bias (8)─┴────────┤
                  │                     │
                  └─────────────────────┴──► Softmax ──► Output
```

### Execution with 4 Threads

```c
// From ggml.c:20460
ggml_graph_compute_thread() {
    
    for (int node_n = 0; node_n < 3; node_n++) {
        
        // NODE 0: MatMul
        ggml_compute_forward(&params, MatMul_node);
        //   ├─ n_tasks = 4 (from ggml_get_n_tasks)
        //   ├─ Thread 0: rows 0-3
        //   ├─ Thread 1: rows 4-7
        //   ├─ Thread 2: rows 8-11
        //   └─ Thread 3: rows 12-15
        
        ggml_barrier();  // All threads wait for MatMul to complete
        
        
        // NODE 1: Add
        ggml_compute_forward(&params, Add_node);
        //   ├─ n_tasks = 4 (from ggml_get_n_tasks)
        //   ├─ Thread 0: elements 0-3
        //   ├─ Thread 1: elements 4-7
        //   ├─ Thread 2: elements 8-11
        //   └─ Thread 3: elements 12-15
        
        ggml_barrier();  // All threads wait for Add to complete
        
        
        // NODE 2: Softmax
        ggml_compute_forward(&params, Softmax_node);
        //   ├─ n_tasks = MIN(4, 8) = 4 (from ggml_get_n_tasks)
        //   ├─ Thread 0: row 0
        //   ├─ Thread 1: row 1
        //   ├─ Thread 2: row 2
        //   └─ Thread 3: row 3
        //   (Only 4 rows, so 4 threads are enough)
        
        ggml_barrier();  // All threads wait for Softmax to complete
    }
}
```

### Parallelization Pattern
```
All nodes (MatMul, Add, Softmax) → All threads process all nodes
                                 ↓
Within each node → Work divided across threads based on operation type
                ↓
Each thread → Processes its assigned work for that node
           ↓
Barrier → Synchronize before next node
```

---

## Example 4: Performance Analysis - 16x512 MatMul

### Setup
```
Weight: 512 x 256 (512 rows, 256 columns)
Activation: 256 x 1
Output: 512 x 1

With 4 threads (RPi4) vs 8 threads (Desktop)
```

### Execution Timeline

**With 4 threads:**
```
Range per thread = 512 / 4 = 128 rows

Thread 0: Compute rows 0-127   Time: 128*256 ops ≈ T₁
Thread 1: Compute rows 128-255 Time: 128*256 ops ≈ T₁
Thread 2: Compute rows 256-383 Time: 128*256 ops ≈ T₁
Thread 3: Compute rows 384-511 Time: 128*256 ops ≈ T₁

Total time: ~T₁ (perfect parallelization if balanced)
Speedup: ~4x (compared to single thread)
```

**With 8 threads:**
```
Range per thread = 512 / 8 = 64 rows

Thread 0: Compute rows 0-63   Time: 64*256 ops ≈ T₁/2
Thread 1: Compute rows 64-127 Time: 64*256 ops ≈ T₁/2
...
Thread 7: Compute rows 448-511 Time: 64*256 ops ≈ T₁/2

Total time: ~T₁/2 (if 8 cores available)
Speedup: ~8x (compared to single thread)
```

### Block-Tiling Impact

Without tiling: Single thread processes all 512*256 = 131k vec_dot calls
└─ Cache misses for most accesses (row data not in L2)

With 16x16 block tiling:
- Thread 0 processes 128 rows in 16-row chunks
  - Each row accessed 16 times before context switch
  - Row data stays in L2 cache (typical L2 = 256KB per core)
  - 16x reduction in effective memory bandwidth needed

---

## Example 5: Work Buffer Usage

### Scenario: BitNet MatMul with quantized weights

```c
// Backend allocation (ggml-backend.cpp:942)
cplan.work_size = ggml_bitnet_mul_mat_get_wsize(src0, src1, dst);
//                ≈ 512*256/8 + 256*4 + 512*4  = 16KB + 1KB + 2KB ≈ 19KB

cplan.work_data = malloc(cplan.work_size + CACHE_LINE_SIZE * 4);
//                 Allocate with padding to prevent false-sharing
```

### Buffer Layout
```
cplan.work_data:

[0-16KB]        Quantized weight cache (512 x 256 / 8 = 16KB)
[16KB-17KB]     Preprocessed activations (256 x 4 = 1KB)
[17KB-19KB]     Output accumulator buffers (512 x 4 = 2KB)
[19KB-19.25KB]  Cache line padding for Thread 0
[19.25KB-19.5KB] Cache line padding for Thread 1
[19.5KB-19.75KB] Cache line padding for Thread 2
[19.75KB-20KB]  Cache line padding for Thread 3

Total: ~20KB per inference
```

### Memory Access Pattern (Thread 0)
```
1. Read weight cache: [0-2KB]    (512 rows / 4 threads = 128 rows)
2. Read activations: [16KB-17KB] (shared by all threads)
3. Write to output:  [17KB-17.5KB] (128 rows / 4 * 4 bytes per row)
4. Use padding: [19KB-19.25KB]   (avoid false-sharing with other threads)

L1 cache (32KB per core): Can hold weight chunk + activations
L2 cache (256KB per core): Can hold full weight + activations
L3 cache (4MB): Shared, not critical
```

---

## Key Takeaways

1. **All threads process all nodes** - Not task-based queue
2. **Work divided within each operation** - Based on `params.ith` and `params.nth`
3. **MatMul work = row chunks** - Thread `i` computes rows `[i*M/N, (i+1)*M/N)`
4. **Synchronization points** - Barriers between nodes ensure correctness
5. **BitNet preprocessing** - Single-threaded but amortized (done once per column)
6. **Block tiling** - 16x16 blocks for cache optimization
7. **NUMA awareness** - Thread affinity for NUMA systems
8. **OpenMP or custom** - Two parallelization strategies, same logic
