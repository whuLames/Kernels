#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>((&pointer))[0])  // 向量化读取
#define OFFSET(row, col, ld) (row * ld + col) // 计算offset

// a naive implementation of sgemm
__global__ void naive_gemm(float *A, float *B, float *C, int M, int K, int N) {
    // a thread for a element in C
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_element = M * N;
    if (tid < out_element) {
        int i = tid / N;  // row id
        int j = tid % N;  // col id

        float sum = 0.0f;
        for(int k = 0; k < K; k ++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[tid] = sum;
    }
}


// a implementation of sgemm with split optmization used for shared memory and reg
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
    // 1. 每个block 搬运数据到 shared memory
    // 2. 每个thread 搬运数据到 reg 并进行计算
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M]; // 为了后续的向量化顺序读取
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    int THREAD_NUM_X_PER_BLOCK = (BLOCK_SIZE_N / THREAD_SIZE_X);
    int THREAD_NUM_Y_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_Y);
    int tid = by * THREAD_NUM_X_PER_BLOCK + bx;
    int THREAD_NUM_PER_BLOCK = THREAD_NUM_X_PER_BLOCK * THREAD_NUM_Y_PER_BLOCK;

    int ITER_PER_BLOCK = K / BLOCK_SIZE_K; // shared memory层面 大迭代的次数 
    int ITER_PER_THREAD = BLOCK_SIZE_K; // register 层面 小迭代的次数

    int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int A_TILE_THREAD_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    int B_TILE_THREAD_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    int A_TILE_THREAD_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    int B_TILE_THREAD_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    int ldg_a_num = (BLOCK_SIZE_M * BLOCK_SIZE_K) / 4 / THREAD_NUM_PER_BLOCK;
    int ldg_b_num = (BLOCK_SIZE_K * BLOCK_SIZE_N) / 4 / THREAD_NUM_PER_BLOCK;

    float ldg_a_reg[4 * ldg_a_num];
    float ldg_b_reg[4 * ldg_b_num];

    float reg_a[THREAD_SIZE_Y];
    float reg_b[THREAD_SIZE_X];

    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0}; // store the thread res

    // 直接给出偏移
    A = &A[BLOCK_SIZE_M * by * K]; // BLOCK_SIZE_M * by 相当于row id
    B = &B[BLOCK_SIZE_N * bx];

    for(int iter = 0; iter < ITER_PER_BLOCK; iter ++) { // 遍历 M*N 个block
        // load data from global memory to shared_memory
        // load A block
        for(int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
            // row_id: A_TILE_THREAD_ROW_START + i
            // col_id: A_TILE_THREAD_COL
            int lda_index = (i / A_TILE_ROW_STRIDE) * 4;
            FETCH_FLOAT4(ldg_a_reg[lda_index]) = FETCH_FLOAT4(A[OFFSET(  // load data from global memory to register
                A_TILE_THREAD_ROW_START + i,
                A_TILE_THREAD_COL + BLOCK_SIZE_K * iter,
                K
            )]);

            // As[i+A_TILE_THREAD_ROW_START][A_TILE_THREAD_COL+0] = ldg_a_reg[lda_index + 0];
            // As[i+A_TILE_THREAD_ROW_START][A_TILE_THREAD_COL+1] = ldg_a_reg[lda_index + 1];
            // As[i+A_TILE_THREAD_ROW_START][A_TILE_THREAD_COL+2] = ldg_a_reg[lda_index + 2];
            // As[i+A_TILE_THREAD_ROW_START][A_TILE_THREAD_COL+3] = ldg_a_reg[lda_index + 3];

            As[A_TILE_THREAD_COL+0][i+A_TILE_THREAD_ROW_START] = ldg_a_reg[lda_index + 0];
            As[A_TILE_THREAD_COL+1][i+A_TILE_THREAD_ROW_START] = ldg_a_reg[lda_index + 1];
            As[A_TILE_THREAD_COL+2][i+A_TILE_THREAD_ROW_START] = ldg_a_reg[lda_index + 2];
            As[A_TILE_THREAD_COL+3][i+A_TILE_THREAD_ROW_START] = ldg_a_reg[lda_index + 3];
        }

        // load B block
        for(int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            // row_id: A_TILE_THREAD_ROW_START + i
            // col_id: A_TILE_THREAD_COL
            int lda_index = (i / B_TILE_ROW_STRIDE) * 4;
            FETCH_FLOAT4(ldg_b_reg[lda_index]) = FETCH_FLOAT4(B[OFFSET(  // load data from global memory to register
                B_TILE_THREAD_ROW_START + i + BLOCK_SIZE_K * iter,
                B_TILE_THREAD_COL,
                N
            )]);

            Bs[i + B_TILE_THREAD_ROW_START][B_TILE_THREAD_COL+0] = ldg_a_reg[lda_index+0];
            Bs[i + B_TILE_THREAD_ROW_START][B_TILE_THREAD_COL+1] = ldg_a_reg[lda_index+1];
            Bs[i + B_TILE_THREAD_ROW_START][B_TILE_THREAD_COL+2] = ldg_a_reg[lda_index+2];
            Bs[i + B_TILE_THREAD_ROW_START][B_TILE_THREAD_COL+3] = ldg_a_reg[lda_index+3];
        }
        __syncthreads();

        
        // 每个thread负责一小块元素的计算
        for(int i = 0; i < ITER_PER_THREAD; i ++) {
            // int THREAD_NUM_X_PER_BLOCK = (BLOCK_SIZE_N / THREAD_SIZE_X);
            // int THREAD_NUM_Y_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_Y);

            // __shared__ As[BLOCK_SIZE_M][BLOCK_SIZE_K];
            // __shared__ Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

            // load A data from smem to reg
            for(int thread=0; thread < THREAD_SIZE_Y; thread += 4) {
                int id = thread / 4 * 4;
                // row: ty  col:
                FETCH_FLOAT4(reg_a[thread]) = FETCH_FLOAT4(As[i][ty * THREAD_SIZE_Y + thread]);
            }

            // load B data from smem to reg
            for(int thread=0; thread < THREAD_SIZE_X; thread += 4) {
                int id = thread / 4 * 4;
                FETCH_FLOAT4(reg_b[thread]) = FETCH_FLOAT4(Bs[i][tx * THREAD_SIZE_X + thread]);
            }

            // do computation
            float tmp = 0.0f;
            for(int i = 0; i < THREAD_SIZE_X; i ++) {
                for(int j = 0; j < THREAD_SIZE_Y; j ++) {
                    accum[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
    }

    // write the res to global memory
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
    
}

// a implementation of sgemm with split optimization and prefetch
// template <
// const int BLOCK_SIZE_M,
// const int BLOCK_SIZE_K,
// const int BLOCK_SIZE_N,
// const int THREAD_SIZE_X,
// const int THREAD_SIZE_Y
// >
// __global__ void gemm_v2(
//     float * __restrict__ A,
//     float * __restrict__ B,
//     float * __restrict__ C, 
//     const int M,
//     const int N,
//     const int K)
// {

// }


int main(int agrc, char** argv) {
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    float* A_h = (float*)malloc(M*K*sizeof(float));
    float* B_h = (float*)malloc(K*N*sizeof(float));
    float* C_h = (float*)malloc(N*K*sizeof(float));

    for(int i = 0; i < M * K; i ++) A_h[i] = 1.0f;
    for(int i = 0; i < K * N; i ++) B_h[i] = 1.0f;
    printf("host data init !\n");

    float* A_d;
    float* B_d;
    float* C_d;

    cudaMalloc((void**)&A_d, M*K*sizeof(float));
    cudaMalloc((void**)&B_d, K*N*sizeof(float));
    cudaMalloc((void**)&C_d, M*N*sizeof(float));
    printf("device data allocate ! \n");

    // memory copy
    cudaMemcpy(A_d, A_h, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K*N*sizeof(float), cudaMemcpyHostToDevice);
    printf("memory copy ! \n");

    // dim3 grid_size((M * N + 256 - 1) / 256);
    // dim3 block_size(256);
    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start);
    // naive_gemm<<<grid_size, block_size>>>(A_d, B_d, C_d, M, K, N);
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // float time;
    // cudaEventElapsedTime(&time, start, end);
    // printf("naive gemm time: %f\n", time);

    // cudaEventRecord(start);
    // naive_gemm<<<grid_size, block_size>>>(A_d, B_d, C_d, M, K, N);
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&time, start, end);
    // printf("naive gemm time: %f\n", time);
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);


    // 每个block的共享内存: (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_N * BLOCK_SIZE_K) * 4 bytes
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    dim3 grid_size(BLOCK_SIZE_N/THREAD_SIZE_X, BLOCK_SIZE_M/THREAD_SIZE_Y);
    dim3 block_size(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
}