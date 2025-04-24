#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
using namespace std;
#define THREAD_NUM_PER_BLOCK 256 


__device__ float warpReduce(float val)
{
    for(int offset = 16; offset > 0; offset >>=1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ float blockReduce(float val)
{
    __shared__ float warpRes[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int tid = threadIdx.x;
    val = warpReduce(val);
    if(lane_id == 0) {
        warpRes[warp_id] = val;
        // printf("warpval %f \n", val);
    }
    __syncthreads();

    float finalval = tid < 32 ? warpRes[tid] : 0;
    // if(tid==0) printf("blockval %f \n", finalval);
    finalval = warpReduce(val); 
    //if(tid==0) printf("blockval %f \n", finalval);
    return finalval;
}


template <unsigned int blockSize>
__global__ void reduce_0(float* in, float* out)
{
    // 一个block处理256个元素
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid * blockSize + tid;
    
    float val = in[idx] + in[idx + blockDim.x];

    val = blockReduce(val);
    // if(tid == 0) printf("bid %d val is %f\n", bid, val);
    if(tid==0) out[bid] = val;  // write back the result
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
    reduce_0<THREAD_NUM_PER_BLOCK><<<gridSize, blockSize>>>(inGPU, outGPU);
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