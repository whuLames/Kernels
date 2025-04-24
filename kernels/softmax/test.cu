#include <stdio.h>
#include <cuda_runtime.h>

// used for
struct __align__(8) MS {float m; float s;};

__device__ MS warpReduce(MS val)
{
    for(int offset = 16; offset > 0; offset /= 2)
    {
        MS tmp;
        tmp.m = __shfl_xor_sync(0xFFFFFFFF, val.m, offset);
        tmp.s = __shfl_xor_sync(0xFFFFFFFF, val.s, offset);

        bool is_val_bigger = val.m > tmp.m;
        MS bigger = is_val_bigger ? val : tmp;
        MS smaller = is_val_bigger ? tmp : val;

        val.m = bigger.m;
        val.s = bigger.s + smaller.s * expf(smaller.m - bigger.m); // 更小的项需要添加偏移量
    }
    return val;
}

__device__ MS reduce2MS(MS x, MS y)
{
    bool is_bigger = x.m > y.m;
    MS bigger = is_bigger ? x : y;
    MS smaller = is_bigger ? y : x;
    
    MS tmp;
    tmp.m = bigger.m;
    tmp.s = bigger.s + smaller.s * expf(smaller.m - bigger.m);
    
    return tmp;
}
__global__ void online_softmax(float* data, float* out, int N, int C)
{
    // oneline softmax kernel
    // The shape of data is N * C
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    MS thread_res = {-INFINITY, 0}; 
    
    while(tid < C)
    {
        float new_val = data[bid * C + tid];
        float new_max_val = fmax(thread_res.m, new_val);
        thread_res.s = thread_res.s * expf(thread_res.m - new_max_val) + expf(new_val - new_max_val);
        thread_res.m = new_max_val;
        tid += blockDim.x;
    }

    __shared__ MS warpRes[32];
    MS thread_res = warpReduce(thread_res);
    if(tid % 32 == 0) {
        warpRes[tid / 32] = thread_res;
    }
    __syncthreads();
    MS empty = {0, 0};
    MS thread_res = tid < 32 ? warpRes[tid] : empty;
    MS block_res = warpReduce(thread_res);

    __shared__ MS blockRes;
    if(tid == 0) {
        // every block write back the res to smem
        blockRes = block_res;
    }

    __syncthreads();
    // the final pass
    for(int i = tid; i < C; i += blockDim.x)
    {
        int idx = bid * C + tid;
        out[idx] = expf(data[idx] - blockRes.m) / blockRes.s;
    }
}

template<int BLOCK_SIZE>  // assume the BLOCK_SIZE is 32
__global__ void transpose(float* data, float* out, int M, int N)
{
    // transpose kernel uses smem and avoid bank conflict
    // The shape of input matrix is M * N
    // The shape of output matrix is N * M
    // grid [N / BLOCKSIZE, M / BLOCKSIZE]  block [BLOCK_SIZE, BLOCK_SIZE]
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1]; // padding 
    int row_A = by * blockDim.y + ty;
    int col_A = bx * blockDim.x + tx;
    if(row_A < M && col_A < N) {
        smem[ty][tx] = data[row_A * N + col_A]; // 合并访问
    }
    
    int row_B = bx * blockDim.x + ty;
    int col_B = by * blockDim.y + tx;
    if(row_B < N && col_B < M) {
        data[row_B * M + col_B] = smem[tx][ty]; // 合并访问
    }
}

template <
const int BLOCK_SIZE_M,
const int BLOCK_SIZE_K,
const int BLOCK_SIZE_N,
const int THREAD_SIZE_X,
const int THREAD_SIZE_Y
>
__global__ void gemm_v1(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K)
{
    // a tiling gemm kernel 
}