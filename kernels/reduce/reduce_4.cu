#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
using namespace std;
#define THREAD_NUM_PER_BLOCK 256 
/*
    The optimization of the reduce_3 is to avoid the synchroization of the last iteration.
 */

__device__ void warp_reduce(volatile float* cache, int tid)
{
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}
// The kernel function of one-iter reduce
__global__ void reduce_0(float* in, float* out)
{
    // shared memory
    __shared__ float sdata[THREAD_NUM_PER_BLOCK];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x * 2 + tid;

    // load data to shared memory
    sdata[tid] = in[id] + in[id + blockDim.x];
    __syncthreads();

    // reduce
    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if(tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if(tid < 32) warp_reduce(sdata, tid);
    // write the result to global memory
    if(tid == 0) out[blockIdx.x] = sdata[0];
}

int main(int argc, char** argv)
{
    // The number of element to reduce
    int N = argc == 2 ? atoi(argv[1]) : 32 * 1024 * 1024;
    cout << "N: " << N << endl;
    int blockNum = (N + 2*THREAD_NUM_PER_BLOCK - 1) / (2*THREAD_NUM_PER_BLOCK);
    cout << "blockNum: " << blockNum << endl;

    // memory malloc
    float* inCPU = (float*)malloc(N * sizeof(float));
    float* outCPU = (float*)malloc(blockNum * sizeof(float));
    float* inGPU;
    float* outGPU;

    cudaMalloc((void**)&outGPU, blockNum * sizeof(float));
    cudaMalloc((void**)&inGPU, N * sizeof(float));
    
    // data init
    for(int i = 0; i < N; i ++) inCPU[i] = 1.0f;
    
    float* ans = (float*)malloc(blockNum * sizeof(float));

    for(int i = 0; i < blockNum; i ++) ans[i] = 0.0f;

    // build the reduce result for correctness check
    for(int i = 0; i < blockNum; i ++)
    {
        for(int j = 0; j < 2 * THREAD_NUM_PER_BLOCK; j ++)
        {
            int idx = i * THREAD_NUM_PER_BLOCK * 2 + j;
            if(idx < N) ans[i] += inCPU[idx];
        }
    }

    // memory copy
    cudaMemcpy(inGPU, inCPU, N * sizeof(float), cudaMemcpyHostToDevice);

    
    dim3 gridSize(blockNum, 1);
    dim3 blockSize(THREAD_NUM_PER_BLOCK, 1);

    // time record
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*
    The optimization of the reduce_3 is to deal with idle threads.
    */
    
    cudaEventRecord(start, 0);
    // computation 
    reduce_0<<<gridSize, blockSize>>>(inGPU, outGPU);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "The number of reduced element: " << N << " Elapsed Time: " << elapsedTime << "ms" << endl;


    // memory copy
    cudaMemcpy(outCPU, outGPU, blockNum * sizeof(float), cudaMemcpyDeviceToHost);

    // check the correctness
    for(int i = 0; i < blockNum; i ++)
    {
        if(outCPU[i] != ans[i])
        {
            cout << "Error: " << i << " " << outCPU[i] << " " << ans[i] << endl;
        }
    }

    return 0;

}