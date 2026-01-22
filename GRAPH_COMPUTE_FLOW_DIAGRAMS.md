# CPU Backend Computation Flow Diagram

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer                                               │
│ (run_inference.py, llama.cpp)                                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Backend Interface Layer (ggml-backend.cpp)                      │
│                                                                  │
│ ggml_backend_cpu_graph_compute()                                │
│ ├─ Allocates work buffer                                         │
│ └─ Calls ggml_graph_compute()                                    │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Computation Planning Layer (ggml.c)                             │
│                                                                  │
│ ggml_graph_plan() ◄─── Analyzes graph                            │
│ └─ ggml_get_n_tasks() ◄─── Determines task count per operation  │
│                                                                  │
│ For MUL_MAT: n_tasks = n_threads                                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Thread Orchestration Layer (ggml.c)                             │
│                                                                  │
│ ggml_graph_compute()                                             │
│ ├─ Creates/initializes threadpool                                │
│ ├─ (OpenMP) #pragma omp parallel                                │
│ │  └─ Each thread calls ggml_graph_compute_thread()             │
│ └─ (No OpenMP) ggml_graph_compute_kickoff()                     │
│    └─ Wakes worker threads                                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Per-Thread Computation Layer (ggml.c)                           │
│                                                                  │
│ ggml_graph_compute_thread(worker_state)                         │
│ ├─ For each node in graph:                                       │
│ │  ├─ ggml_compute_forward(&params, node)                       │
│ │  │  └─ Dispatches to operation-specific function              │
│ │  └─ ggml_barrier() [Wait for all threads]                     │
│ └─ Return                                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Operation Dispatch Layer (ggml.c)                               │
│                                                                  │
│ ggml_compute_forward() [Switch on operation type]               │
│ ├─ GGML_OP_MUL_MAT       ──► ggml_compute_forward_mul_mat()      │
│ ├─ GGML_OP_ADD           ──► ggml_compute_forward_add()          │
│ ├─ GGML_OP_ROPE          ──► ggml_compute_forward_rope()         │
│ └─ ... [other operations]                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Matrix Multiply Implementation Layer (ggml.c)                   │
│                                                                  │
│ ggml_compute_forward_mul_mat()                                   │
│ ├─ Check for BitNet optimization (ARM TL1 / X86 TL2)            │
│ │  ├─ ggml_bitnet_can_mul_mat() ──► YES                         │
│ │  │  └─ ggml_qgemm_lut() [Quantized GEMM with LUT]            │
│ │  └─ NO                                                         │
│ │     └─ ggml_compute_forward_mul_mat_one_chunk()               │
│ └─ Each thread computes row range [ith*M/nth, (ith+1)*M/nth]    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Matrix Multiply Thread Distribution

```
Input Matrices:
  Weight (M x K)        Activation (K x N)
  ┌───────────────────┐ ┌─────────────────┐
  │ w0  w1  ... wK-1  │ │ a0 a1 ... aN-1  │
  │ w0  w1  ... wK-1  │ │ a0 a1 ... aN-1  │
  │ ...               │ │ ...             │
  │ w0  w1  ... wK-1  │ │ a0 a1 ... aN-1  │
  └───────────────────┘ └─────────────────┘
  (M rows, K cols)     (K rows, N cols)

Output Matrix (M x N):
  ┌──────────────────────────────────────┐
  │ r00 r01 r02 ... r0(N-1)              │ ◄─ Thread 0: rows 0 to M/nth
  │ r10 r11 r12 ... r1(N-1)              │
  │ ...                                  │
  │ r(M/nth-1)0 ... ... r(M/nth-1)(N-1)  │ ◄─ All threads work independently
  ├──────────────────────────────────────┤
  │ r(M/nth)0 r(M/nth)1 ... ...          │ ◄─ Thread 1: rows M/nth to 2*M/nth
  │ ...                                  │
  │ r(2*M/nth-1)0 ... ... ...            │
  ├──────────────────────────────────────┤
  │ ...                                  │
  │                                      │
  │ r((nth-1)*M/nth)0 ... ... r(M-1)(N-1)│ ◄─ Thread nth-1: rows (nth-1)*M/nth to M
  └──────────────────────────────────────┘
```

---

## Parallel Execution Timeline (for 4 threads, 3-node graph)

```
Timeline ─────────────────────────────────────────────────────────────►

Node 1: Add (parallelizable)
  Thread 0: [======= Add computation =======]
  Thread 1: [======= Add computation =======]
  Thread 2: [======= Add computation =======]
  Thread 3: [======= Add computation =======]
           ▲                                ▲
           └────────────────────────────────┘
                   All threads synchronized (Barrier)

Node 2: MatMul (highly parallelizable - our focus)
  Thread 0: [== MatMul rows 0:M/4 ==]
  Thread 1: [== MatMul rows M/4:M/2 ==]
  Thread 2: [== MatMul rows M/2:3M/4 ==]
  Thread 3: [== MatMul rows 3M/4:M ==]
           ▲                              ▲
           └──────────────────────────────┘
                   All threads synchronized (Barrier)

Node 3: Softmax (limited parallelization)
  Thread 0: [============ Softmax ============]
  Thread 1: [============ Softmax ============]
  Thread 2: [============ Softmax ============]
  Thread 3: [============ Softmax ============]
           ▲                                ▲
           └────────────────────────────────┘
                   All threads synchronized (Barrier)
```

---

## BitNet Quantized MatMul Work Division (ARM TL1)

```
For: Weight (M x K, quantized) @ Activation (K x N) ──► Output (M x N)

BitNet Parameters:
  • Weights: 1-bit or ternary quantized
  • LUT: Lookup table for efficient computation
  • Scales: Per-row/group quantization scales

Execution with N threads:

  Main thread (ith = 0):
  ┌─────────────────────────────────────────┐
  │ ggml_bitnet_transform_tensor()          │ (Single-threaded preprocessing)
  │ ggml_preprocessor()                     │ (Convert activations to int form)
  └──────────────────┬──────────────────────┘
                     │
                     ▼
              BARRIER (wait all threads)
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
  [Thread 0]   [Thread 1]   [Thread N-1]
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Compute  │ │ Compute  │ │ Compute  │
  │ rows:    │ │ rows:    │ │ rows:    │
  │ 0 to M/N │ │ M/N to   │ │ (N-1)*M/N│
  │ using    │ │ 2*M/N    │ │ to M     │
  │ ggml_    │ │ using    │ │ using    │
  │ qgemm_   │ │ ggml_    │ │ ggml_    │
  │ lut()    │ │ qgemm_   │ │ qgemm_   │
  │          │ │ lut()    │ │ lut()    │
  └──────────┘ └──────────┘ └──────────┘
         │           │           │
         └───────────┼───────────┘
                     │
                     ▼
              BARRIER (wait all threads)
```

---

## Memory Layout - Work Buffer

```
Work Buffer (allocated in ggml_backend_cpu_graph_compute):

┌────────────────────────────────────────────────────┐
│  Row 0              │  Row K-1             │       │
│  Transformation     │  (quantized weights) │       │
│  Space              │                      │       │
├────────────────────────────────────────────────────┤
│                 Quantized Weights Cache             │
│            (for type conversion if needed)          │
├────────────────────────────────────────────────────┤
│            BitNet Kernel Output Cache               │
│         (for intermediate quantized results)        │
├────────────────────────────────────────────────────┤
│                 Thread-Local Buffers                │
│  (CACHE_LINE_SIZE * n_threads for avoid false-sharing)
└────────────────────────────────────────────────────┘
  ▲                                                   ▲
  └─────────────────────────────────────────────────┘
         cplan.work_data pointer
         Size calculated in ggml_graph_plan()
```

---

## Operation Type → Task Count Mapping

```
┌─────────────────────────────────────────────────────────┐
│ Operation Type          │ Task Count (n_tasks)          │
├─────────────────────────────────────────────────────────┤
│ GGML_OP_MUL_MAT        │ n_threads     [FULL PARALLEL] │
│ GGML_OP_MUL_MAT_ID     │ n_threads     [FULL PARALLEL] │
│ GGML_OP_OUT_PROD       │ n_threads     [FULL PARALLEL] │
│ GGML_OP_ADD            │ n_threads     [FULL PARALLEL] │
│ GGML_OP_ROPE           │ n_threads     [FULL PARALLEL] │
│ GGML_OP_FLASH_ATTN_EXT │ n_threads     [FULL PARALLEL] │
│ GGML_OP_SSM_CONV       │ n_threads     [FULL PARALLEL] │
├─────────────────────────────────────────────────────────┤
│ GGML_OP_SOFT_MAX       │ min(n_threads, rows) [LIMITED]│
├─────────────────────────────────────────────────────────┤
│ GGML_OP_SUM            │ 1             [SINGLE]        │
│ GGML_OP_MEAN           │ 1             [SINGLE]        │
│ GGML_OP_ARGMAX         │ 1             [SINGLE]        │
│ GGML_OP_VIEW           │ 1             [SINGLE]        │
│ GGML_OP_PERMUTE        │ 1             [SINGLE]        │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow in MatMul Computation

```
For a single output element dst[i][j]:

1. Input Setup
   ┌─────────────────────────────────────┐
   │ src0[i][0..K-1] (weight row i)      │
   │ src1[j][0..K-1] (activation row j)  │
   └─────────────────────────────────────┘

2. Thread-based Work Division
   ├─ Thread assigns based on: ith = params->ith
   ├─ Row range: [ith * (M/nth), (ith+1) * (M/nth)]
   └─ Each thread processes its assigned rows for all columns

3. Computation Path (BitNet ARM TL1)
   ┌──────────────────────────────────────┐
   │ vec_dot(src0_row, src1_col) via LUT  │
   │ • Quantized weight lookup            │
   │ • Integer multiply-accumulate        │
   │ • Quantization scaling (via scales)  │
   └─────────────────┬────────────────────┘
                     ▼
   ┌──────────────────────────────────────┐
   │ Dequantize result and store          │
   │ dst[i][j] = result / act_scale * scale
   └──────────────────────────────────────┘

4. Synchronization
   └─ All threads barrier before next node
```

---

## Cache-Aware Block Tiling in Non-BitNet MatMul

```
Standard MatMul uses block-tiling for cache efficiency:

Output Matrix divided into 16x16 blocks:

┌─────────────────────────────────────┐
│ [0,0]  [0,1]  [0,2]  [0,3] ...      │
│ [1,0]  [1,1]  [1,2]  [1,3] ...      │
│ [2,0]  [2,1]  [2,2]  [2,3] ...      │  ◄─ 16x16 block
│ ...  ┌──────────────┐                │
│ [15,0]│ [15,1]│ ...  │ [15,15]       │
│ ───────┴──────────────┴──────────    │
│ [16,0]  [16,1]  [16,2]  [16,3] ...  │
│ ...                                 │
└─────────────────────────────────────┘

Benefits:
• L2 cache reuse: Each source row accessed 16 times
• Reduced memory bandwidth: ~16x improvement
• Each thread processes blocks independently
```
