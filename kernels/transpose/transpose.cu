#include <stdio.h>
#include <cuda_runtime.h>


__global__ void transpose_v0(float *data, float* out, int M, int N)
{
    // gridDim.x = M / blockDim.x
    // gridDim.y = N / blockDim.y
    // 这里的row 和 column 是转置后的矩阵的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < M)
    {
        // data 读取连续
        // out 读取不连续
        out[col * N + row] = data[row * M + col];
    }
}

template<int BLOCK_SIZE>
__global__ void transpose_v1(float* data, float* out, int M, int N)
{
    // 避免由smem引起的 bank conflict问题

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /*
    核心在于直接读写的方式必然会造成读取或写入不能合并的情况
    所以采用shared_memory进行中转，值得注意的是, 对shared_memory的读写不用考虑连续读取的问题
    */
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE];
    int row_A = by * BLOCK_SIZE + ty;
    int col_A = bx * BLOCK_SIZE + tx;

    if(row_A < M && col_A < N) {  // BLOCK_SIZE * BLOCK_SIZE的矩阵平移
        smem[ty][tx] = data[row_A * N + col_A];
    }

    int row_B = bx * BLOCK_SIZE + ty;
    int col_B = by * BLOCK_SIZE + tx; // 按行读取
    if(row_B < N && col_B < M) {
        out[row_B * M + col_B] = smem[tx][ty];
    }
}

template <int BLOCK_SIZE>
__global__ void transpose_v2(float *data, float* out, int M, int N)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /*
    核心在于直接读写的方式必然会造成读取或写入不能合并的情况
    所以采用shared_memory进行中转，值得注意的是, 对shared_memory的读写不用考虑连续读取的问题
    */
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE+1];
    int row_A = by * BLOCK_SIZE + ty;
    int col_A = bx * BLOCK_SIZE + tx;

    if(row_A < M && col_A < N) {  // BLOCK_SIZE * BLOCK_SIZE的矩阵平移
        smem[ty][tx] = data[row_A * N + col_A];  
    }

    int row_B = bx * BLOCK_SIZE + ty;
    int col_B = by * BLOCK_SIZE + tx; // 按行读取
    if(row_B < N && col_B < M) {
        out[row_B * M + col_B] = smem[tx][ty];  // 此时存在16-way bank conflict
    }
    // 假设线程之间访问数据的偏移量为 offset
    // 如果offset 和 32 互质, 那么不存在bank conflict
    // 如果不互质, 即最小公倍数为 a, 则 x-way 冲突的 x = 32 / (a / offset) = 32 * offset  / a

    
}



template<int BLOCK_SIZE>
__global__ void transpose_v3(float* data, float* out, int M, int N)
{
    // 避免由smem引起的 bank conflict问题

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    /*
    核心在于直接读写的方式必然会造成读取或写入不能合并的情况
    所以采用shared_memory进行中转，值得注意的是, 对shared_memory的读写不用考虑连续读取的问题
    */
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE];
    int row_A = by * BLOCK_SIZE + ty;
    int col_A = bx * BLOCK_SIZE + tx;

    if(row_A < M && col_A < N) {  // BLOCK_SIZE * BLOCK_SIZE的矩阵平移
        smem[ty][tx^ty] = data[row_A * N + col_A];
    }

    int row_B = bx * BLOCK_SIZE + ty;
    int col_B = by * BLOCK_SIZE + tx; // 按行读取
    if(row_B < N && col_B < M) {
        out[row_B * M + col_B] = smem[tx][ty^tx];
    }
}

int main(int argc, char** argv)
{
    const int M = 12800;
    const int N = 1280;
    const int BLOCK_SIZE=32;
    float* h_matrix = (float*)malloc(sizeof(float) * M * N);
    // 初始化数据
    for(int i = 0; i < M * N; i++)
    {
        h_matrix[i] = i % N;
    }

    float* d_matrix;
    float* d_out;
    cudaMalloc(&d_matrix, sizeof(float) * M * N);
    cudaMalloc(&d_out, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transpose_v0<<<grid, block>>>(d_matrix, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive transpose v0: %f\n", milliseconds);

    cudaEventRecord(start);
    transpose_v1<BLOCK_SIZE><<<grid, block>>>(d_matrix, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("combinated reading rev1: %f\n", milliseconds);
    
    cudaEventRecord(start);
    transpose_v2<BLOCK_SIZE><<<grid, block>>>(d_matrix, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("avoid bc by padding v2: %f\n", milliseconds);

    cudaEventRecord(start);
    transpose_v3<BLOCK_SIZE><<<grid, block>>>(d_matrix, d_out, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("avoid bc by swizzling v3: %f\n", milliseconds);



}