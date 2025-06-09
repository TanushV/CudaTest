#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Simple CUDA kernel to compute pairwise distances between N 3D points
__global__ void pairwiseDistances(const float *points, float *distMatrix, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // column index

    if (i < n && j < n) {
        float3 pi = make_float3(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        float3 pj = make_float3(points[3 * j], points[3 * j + 1], points[3 * j + 2]);

        float dx = pi.x - pj.x;
        float dy = pi.y - pj.y;
        float dz = pi.z - pj.z;

        distMatrix[i * n + j] = sqrtf(dx * dx + dy * dy + dz * dz);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <num_points>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    size_t pointsSize = sizeof(float) * 3 * n;
    size_t matrixSize = sizeof(float) * n * n;

    float *h_points = (float *)malloc(pointsSize);
    float *h_matrix = (float *)malloc(matrixSize);

    // Initialize points with some values, here we just use sequential numbers
    for (int i = 0; i < n; ++i) {
        h_points[3*i] = (float)i;
        h_points[3*i+1] = (float)i * 0.5f;
        h_points[3*i+2] = (float)i * 0.25f;
    }

    float *d_points = nullptr;
    float *d_matrix = nullptr;
    cudaMalloc(&d_points, pointsSize);
    cudaMalloc(&d_matrix, matrixSize);

    cudaMemcpy(d_points, h_points, pointsSize, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y);
    pairwiseDistances<<<grid, block>>>(d_points, d_matrix, n);

    cudaMemcpy(h_matrix, d_matrix, matrixSize, cudaMemcpyDeviceToHost);

    // Print first 5x5 block of distances for quick verification
    int display = n < 5 ? n : 5;
    for (int i = 0; i < display; ++i) {
        for (int j = 0; j < display; ++j) {
            printf("%0.2f ", h_matrix[i*n+j]);
        }
        printf("\n");
    }

    cudaFree(d_points);
    cudaFree(d_matrix);
    free(h_points);
    free(h_matrix);
    return 0;
}
