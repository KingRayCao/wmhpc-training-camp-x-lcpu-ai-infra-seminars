#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void saxpy(const float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = 2.0f * x[i] + y[i];
    }
}

struct GpuTimer{
    cudaEvent_t start_e, stop_e;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_e));
        CUDA_CHECK(cudaEventCreate(&stop_e));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_e);
        cudaEventDestroy(stop_e);
    }
    void start() {
        CUDA_CHECK(cudaEventRecord(start_e, 0));
    }
    float stop_ms() {
        CUDA_CHECK(cudaEventRecord(stop_e, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_e, stop_e));
        return ms;
    }
};

int main(int argc, char *argv[]) {
    int n = std::stoi(argv[1]);
    if (n == 0){
        printf("SUM=0, time=0.0 ms, n=0\n");
        return 0;
    }
    size_t bytes = (size_t)n * sizeof(float);
    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (float)((i % 1024) - 512);
    }
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    GpuTimer timer;
    timer.start();
    saxpy<<<blocks, threads>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    float ms = timer.stop_ms();

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += h_y[i];
    printf("SUM=%.0f, time=%.1f ms, n=%d\n", sum, ms, n);

}
