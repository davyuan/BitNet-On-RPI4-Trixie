# CPU Backend Graph Computation - Documentation Index

This documentation package provides comprehensive analysis of how the llama.cpp GGML framework implements CPU-based tensor computation with multi-threaded parallelization.

## 📚 Documentation Files

### 1. **CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md** 
   **Detailed Technical Analysis**
   
   The most comprehensive reference covering:
   - CPU backend entry point and initialization
   - Computation plan creation and analysis
   - Task determination logic for all operation types
   - Main graph computation function with thread orchestration
   - Per-thread computation with synchronization
   - Operation dispatch mechanism
   - Matrix multiplication implementation (both standard and BitNet)
   - Block-tiled optimization strategy
   - Thread synchronization patterns
   - Threadpool management (non-OpenMP)
   - Work distribution patterns
   - BitNet-specific optimizations for quantized computation

   **Best for:** Deep understanding of the complete architecture and flow

---

### 2. **GRAPH_COMPUTE_FLOW_DIAGRAMS.md**
   **Visual Diagrams and Architecture Flows**
   
   Contains:
   - High-level architecture flow diagram (entry point to computation)
   - Matrix multiply thread distribution visualization
   - Parallel execution timeline for multi-node graphs
   - BitNet quantized MatMul work division diagram
   - Memory layout for work buffers
   - Operation type to task count mapping table
   - Data flow for single matrix multiply element
   - Cache-aware block tiling visualization

   **Best for:** Understanding the big picture and visual learners

---

### 3. **QUICK_REFERENCE_CPU_BACKEND.md**
   **Quick Lookup and Code Navigation**
   
   Includes:
   - File locations and key functions table
   - Code navigation guide with call chains
   - Detailed breakdown of 6 critical functions
   - How to find specific features (MatMul, BitNet, threading, sync)
   - Thread parameters structure reference
   - Performance characteristics explanation
   - Configuration options
   - Summary lookup table

   **Best for:** Finding specific code, quick lookups, getting oriented

---

### 4. **PRACTICAL_EXAMPLES_CPU_BACKEND.md**
   **Real-World Examples with Concrete Numbers**
   
   Demonstrates:
   - Example 1: Simple 2D MatMul with 4 threads
   - Example 2: BitNet MatMul with 4 threads (ARM TL1)
   - Example 3: Mixed operations graph execution
   - Example 4: Performance analysis of 16x512 MatMul
   - Example 5: Work buffer usage and memory layout

   **Best for:** Understanding execution through concrete examples

---

## 🎯 Quick Navigation

### If you want to know...

**"How does matrix multiplication get parallelized?"**
→ Start with [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md#ggml_compute_forward_mul_mat---matmul-implementation)
→ Then read [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md#example-1-simple-2d-matrix-multiply-with-4-threads)

**"Where is the actual threaded work happening?"**
→ [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md#7-matmul-implementation---the-real-work) Section 7
→ Find `ggml_compute_forward_mul_mat()` and `ggml_qgemm_lut()`

**"How are threads synchronized?"**
→ [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md#8-thread-synchronization) Section 8
→ See [GRAPH_COMPUTE_FLOW_DIAGRAMS.md](GRAPH_COMPUTE_FLOW_DIAGRAMS.md#parallel-execution-timeline-for-4-threads-3-node-graph)

**"How does BitNet optimization work?"**
→ [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md#11-bitnet-specific-optimizations) Section 11
→ [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md#example-2-bitnet-matmul-with-4-threads-arm-tl1) Example 2

**"What's the performance impact?"**
→ [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md#12-performance-characteristics) Section 12
→ [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md#example-4-performance-analysis---16x512-matmul) Example 4

**"How do I find X in the code?"**
→ [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md#how-to-find-things)

---

## 📊 Key Concepts Summary

### Thread Organization
```
All threads process: ALL graph nodes (not a task queue)
                     ↓
Within each node: Work is divided across threads
                     ↓
Each thread: Processes its assigned work independently
                     ↓
Synchronization: Barrier between nodes ensures correctness
```

### MatMul Work Division
```
For Matrix Multiply (M x K @ K x N):
├─ Output matrix: M x N
├─ Divided by: Number of output rows (M)
├─ Each thread gets: M/N_threads consecutive rows
└─ Each thread computes its rows independently
```

### Key Files
```
ggml-backend.cpp    │ Backend interface, CPU backend initialization
                    ↓
ggml.c (20735)      │ ggml_graph_compute() - Main orchestrator
                    ↓
ggml.c (20460)      │ ggml_graph_compute_thread() - Per-thread loop
                    ↓
ggml.c (17812)      │ ggml_compute_forward() - Operation dispatch
                    ↓
ggml.c (12585)      │ ggml_compute_forward_mul_mat() - MatMul kernel
```

### Synchronization
```
Node 0 computation
        ↓
   BARRIER  ← All threads wait
        ↓
Node 1 computation
        ↓
   BARRIER  ← All threads wait
        ↓
Node 2 computation
```

---

## 🔍 Code Structure Overview

```
3rdparty/llama.cpp/ggml/src/
├── ggml-backend.cpp         [Backend interface]
│   ├─ ggml_backend_cpu_graph_compute()
│   └─ ggml_backend_cpu_graph_plan_create()
│
└── ggml.c                   [Core computation ~24000 lines]
    ├─ ggml_graph_plan()                [Line 20271]
    ├─ ggml_get_n_tasks()               [Line 19790] ← Task count logic
    ├─ ggml_graph_compute()             [Line 20735] ← Main orchestrator
    ├─ ggml_graph_compute_thread()      [Line 20460] ← Per-thread work
    ├─ ggml_graph_compute_secondary_thread() [Line 20569] ← Worker threads
    ├─ ggml_compute_forward()           [Line 17812] ← Dispatch
    ├─ ggml_compute_forward_mul_mat()   [Line 12585] ← MatMul (standard)
    ├─ ggml_compute_forward_mul_mat_one_chunk() [Line 12407] ← Block-tiled
    └─ [... other operations ...]
```

---

## 📋 Checklist for Understanding

- [ ] Read the high-level architecture flow ([GRAPH_COMPUTE_FLOW_DIAGRAMS.md](GRAPH_COMPUTE_FLOW_DIAGRAMS.md) first section)
- [ ] Review the simple MatMul example ([PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md) Example 1)
- [ ] Understand thread parameters and ith/nth ([QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md#thread-parameters-structure))
- [ ] Trace through ggml_graph_compute_thread() function ([CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) Section 5)
- [ ] Study MatMul implementation ([CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) Section 7)
- [ ] Review BitNet example if using quantized models ([PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md) Example 2)
- [ ] Understand synchronization points ([CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) Section 8)

---

## 🎓 Learning Path

### Beginner Path (30 minutes)
1. [GRAPH_COMPUTE_FLOW_DIAGRAMS.md](GRAPH_COMPUTE_FLOW_DIAGRAMS.md) - High-level architecture
2. [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md#example-1-simple-2d-matrix-multiply-with-4-threads) - Example 1
3. [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md#thread-parameters-structure) - Thread parameters

### Intermediate Path (1-2 hours)
1. [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md) - Full reference
2. [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md) - All examples
3. [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) Sections 1-7

### Advanced Path (2-4 hours)
1. [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) - Complete analysis
2. Study the actual code with documentation as reference
3. [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md#example-5-work-buffer-usage) - Example 5
4. Performance analysis sections

---

## 🔗 Important Functions Cross-Reference

| Function | File | Line | Purpose |
|----------|------|------|---------|
| `ggml_backend_cpu_graph_compute()` | ggml-backend.cpp | 942 | Backend entry point |
| `ggml_graph_plan()` | ggml.c | 20271 | Plan computation |
| `ggml_get_n_tasks()` | ggml.c | 19790 | Task count logic |
| `ggml_graph_compute()` | ggml.c | 20735 | Orchestrate threads |
| `ggml_graph_compute_thread()` | ggml.c | 20460 | Per-thread work |
| `ggml_compute_forward()` | ggml.c | 17812 | Dispatch operations |
| `ggml_compute_forward_mul_mat()` | ggml.c | 12585 | Matrix multiply |
| `ggml_compute_forward_mul_mat_one_chunk()` | ggml.c | 12407 | Block-tiled matmul |
| `ggml_barrier()` | ggml.c | N/A | Thread synchronization |
| `ggml_qgemm_lut()` | ggml.c | N/A | BitNet quantized GEMM |

---

## 📝 Notes for Implementation/Debugging

### Understanding Thread Indices
- `params.ith`: Current thread's index (0 to nth-1)
- `params.nth`: Total number of threads
- **Pattern**: Each thread handles work range `[ith * size / nth, (ith+1) * size / nth)`

### Debugging MatMul
1. Check `params.ith` and `params.nth` values
2. Calculate expected row range: `[ith * M / nth, (ith+1) * M / nth)`
3. Verify barrier is called after computation
4. Check for false-sharing with cache line padding

### BitNet-Specific
1. Check if `ggml_bitnet_can_mul_mat()` returns true
2. Verify `ggml_bitnet_transform_tensor()` is called by thread 0 only
3. Ensure preprocessing barrier is hit
4. Check `ggml_qgemm_lut()` parameters match tensor dimensions

### Performance Tuning
1. Block size (typically 16x16) - affects cache locality
2. Thread count - should not exceed physical cores
3. Work distribution - ensure balanced load
4. NUMA affinity - set via `set_numa_thread_affinity()`

---

## 📚 Related Files in Repository

```
/home/david/dev/BitNet-On-RPI4-Trixie/
├── 3rdparty/llama.cpp/
│   └── ggml/src/
│       ├── ggml-backend.cpp         ← CPU backend implementation
│       ├── ggml.c                   ← Core computation engine
│       ├── ggml-bitnet.h            ← BitNet kernel declarations
│       └── ggml-bitnet-lut.cpp      ← BitNet LUT implementation
├── include/
│   ├── ggml-bitnet.h                ← BitNet headers
│   └── bitnet-lut-kernels.h         ← LUT kernel headers
├── src/
│   └── ggml-bitnet-lut.cpp          ← BitNet LUT kernels
└── preset_kernels/                  ← Precompiled BitNet kernels
    ├── bitnet_b1_58-3B/
    ├── bitnet_b1_58-large/
    └── Llama3-8B-1.58-100B-tokens/
```

---

## 💡 Key Insights

1. **All threads execute same code** - Not task-based parallelism, but data-based
2. **Row division** - Matrix multiply divides work by output rows
3. **Fine-grained sync** - Barriers only between nodes, not within node
4. **BitNet advantage** - Quantized computation 8-16x faster than fp32
5. **Cache optimization** - Block tiling (16x16) critical for performance
6. **NUMA aware** - Thread affinity improves NUMA system performance
7. **Two models** - OpenMP (implicit) or custom threadpool (explicit)

---

## 🚀 Getting Started

1. **For quick lookup:** Use [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md)
2. **For understanding flow:** Read [GRAPH_COMPUTE_FLOW_DIAGRAMS.md](GRAPH_COMPUTE_FLOW_DIAGRAMS.md) then [PRACTICAL_EXAMPLES_CPU_BACKEND.md](PRACTICAL_EXAMPLES_CPU_BACKEND.md)
3. **For deep dive:** Study [CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md](CPU_BACKEND_GRAPH_COMPUTE_ANALYSIS.md) with code open
4. **For specific feature:** Use "How to Find Things" in [QUICK_REFERENCE_CPU_BACKEND.md](QUICK_REFERENCE_CPU_BACKEND.md#how-to-find-things)

---

**Generated:** January 21, 2026  
**Repository:** BitNet-On-RPI4-Trixie  
**Codebase:** llama.cpp GGML Framework with BitNet Optimizations  
**Focus:** CPU Backend Graph Computation and Thread Parallelization
