// The implementation of the softmax function with a safe softmax trick.
#include <stdio.h>
#include <cuda_runtime.h>

// the struct for online softmax
struct __align__(8) MS {float m; float s;};

__device__ MS onlineWarpReduce(MS val)
{
    #pragma unroll
    for(int offset=16; offset>0; offset>>=1) {
        MS tmp;
        tmp.m = __shfl_xor_sync(0xFFFFFFFF, val.m, offset);
        tmp.s = __shfl_xor_sync(0xFFFFFFFF, val.s, offset);
        bool val_bigger = val.m > tmp.m;
        MS bigger = val_bigger ? val : tmp;
        MS smaller = val_bigger ? tmp : val;
        val.m = bigger.m;
        val.s = smaller.s * expf(bigger.m - smaller.m) + bigger.s; 
    }

    return val;
}

__device__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while(assumed != old);
    return __int_as_float(old);
}

__device__ float warpMaxReduce(float val)
{
    #pragma unroll
    for(int offset=16; offset>0; offset>>=1) {
        val = max(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
__device__ float warpSumReduce(float val)
{
    #pragma unroll
    for(int offset=16; offset>0; offset>>=1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float blockMaxReduce(float val)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    __shared__ float warpRes[64];
    val = warpMaxReduce(val);
    if(lane_id==0) warpRes[warp_id] = val;
    __syncthreads();
    int tid = threadIdx.x;

    val = tid < 32 ? warpRes[tid] : 0;

    return warpMaxReduce(val);
}

__device__ float blockSumReduce(float val)
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    __shared__ float warpRes[64];
    val = warpSumReduce(val);
    if(lane_id==0) warpRes[warp_id] = val;
    __syncthreads();
    int tid = threadIdx.x;

    val = tid < 32 ? warpRes[tid] : 0;
    return warpSumReduce(val);
}



__global__ void safe_softmax(float* data, float* out, float* max_val, float* sum_val, int N)
{
    // Compute the max value
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float val = data[bid * blockDim.x + tid]; 

    // get the max value of the current block
    float val_in_block = blockMaxReduce(val);

    if(tid == 0)
    {
        atomicMax(max_val, val_in_block);
    }
    __syncthreads();
    
    // Compute the sum of the exp value
    float val_exp = expf(val - *max_val);
    float sum_exp_in_block = blockSumReduce(val_exp);
    if(tid==0)
    {
        atomicAdd(out, sum_exp_in_block); 
    }
    __syncthreads();

    // Compute the softmax value
    out[bid * blockDim.x + tid] = val_exp / *sum_val;
}

__device__ MS reduce2MS(MS x, MS y)
{
    bool x_bigger = x.m > y.m;
    MS bigger = x_bigger ? x : y;
    MS smaller = x_bigger ? y : x;
    x.m = bigger.m;
    x.s = smaller.s * expf(bigger.m - smaller.m) + bigger.s;
    return x;
}
__global__ void online_softmax(float* data, float* out, int N, int C)
{
    // 2-d softmax and one block for a row
    // online softmax --> 2-pass
    // 1st pass: compute the max value and sum of the exp value
    // 2nd pass: compute the softmax value
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    float max_val = -INFINITY;
    float sum_val = 0.0f;
    int stride = blockDim.x;
    MS val = {-INFINITY, 0.0f};
    __shared__ MS warpRes[32];
    __shared__ MS blockRes;
    for(int i = tid; i < C; i += stride)
    {
        float new_val = data[bid * C + i];
        float new_max_val = fmaxf(new_val, max_val);
        val.m = new_max_val;
        val.s = val.s * expf(new_max_val - max_val) + expf(new_val - val.m); // 前者修正 后者加上当前位置的value
    }
    MS res = onlineWarpReduce(val);
    if(lane_id == 0)
    {
        warpRes[warp_id] = res;
    }
    __syncthreads();

    MS val = tid < 32 ? warpRes[tid] : (MS){0.0f, 0.0f};
    MS block_res = onlineWarpReduce(val);
    if(tid == 0)
    {
        blockRes = block_res;
    }
    __syncthreads();

    // 2nd pass
    for(int i = tid; i < C; i += stride)
    {
        out[bid * C + i] = expf(data[bid * C + i] - blockRes.m) / blockRes.s;
    }

}
int main(int argc, char** argv)
{
    const int N = atoi(argv[1]);
    // printf("N = %d\n", N);
    float* h_data = (float*)malloc(N * sizeof(float));
    float* d_data;
    for(int i = 0; i < N; i++)
    {
        h_data[i] = i % 16;
    }
    cudaMalloc((void**)&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(1024);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    float* d_out;
    float* d_max_val;
    float* d_sum_val;
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMalloc((void**)&d_max_val, sizeof(float));
    cudaMalloc((void**)&d_sum_val, sizeof(float));
    cudaMemset(d_out, 0, N * sizeof(float));
    cudaMemset(d_max_val, 0, sizeof(float));
    cudaMemset(d_sum_val, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    safe_softmax<<<gridSize, blockSize>>>(d_data, d_out, d_max_val, d_sum_val, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);

}