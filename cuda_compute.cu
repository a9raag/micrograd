// #include "include/cuda_compute.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


template <typename T>
__global__ void transposeKernel2d(T* data, T* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[j * x + i] = data[i * y + j];
    }
}

template <typename T>
__global__ void fillKernel(T* data, T val, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = val;
    }
}

template <typename T>
__global__ void fillKernel2d(T* data, T val, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        data[i * y + j] = val;
    }
}

template <typename T>
__global__ void powKernel(T* a, T *out, double n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = pow(a[i], n);
    }
}

template <typename T> 
__global__ void powKernel2d(T* a, T* out, double n, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = pow(a[i * y + j], n);
    }
}

__global__ void cudaPowGrad(double* data, double *outGrad, double* grad, double n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += n * pow(data[i], n - 1) * outGrad[i];
    }
}

template <typename T>
__global__ void tanhKernel(T* data, T* out, int size){ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size){
        double x = data[i]; 
        out[i] = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    }
}

template <typename T> 
__global__ void tanhKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y){
        double x = data[i * y + j]; 
        out[i * y + j] = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    }    
}

__global__ void gradTanh(double* grad, double* outGrad, double* data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double t = (exp(2.0 * data[i]) - 1.0) / (exp(2.0 * data[i]) + 1.0);
        grad[i] += (1.0 - pow(t, 2)) * outGrad[i];
    }
}

template<typename T>
__global__ void addKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
__global__ void addKernel(T* a, double b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b;
    }
}

template <typename T>
__global__ void addKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b[i * y + j];
    }
}

template <typename T>
__global__ void addKernel2d(T* a, double b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b;
    }
}

template <typename T> 
__global__ void addKernel2dRowWise(T* a, T* b, T* c, size_t x, size_t y) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b[j];
    }
    
}

template <typename T>
__global__ void addKernel2dColWise(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b[i];
    }
}

__global__ void addGrad(double* grad, double* outGrad, double* otherGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += 1.0 * outGrad[i];
        otherGrad[i] += 1.0 * outGrad[i];
    }
}

template<typename T>
__global__ void mulKernel(T* data, T* other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other[i];
    }
}


template <typename T> 
__global__ void mulKernel(T* data, double other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other;
    }
}

template <typename T>
__global__ void mulKernel2d(T* data, T* other, T* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other[i * y + j];
    }
}

template <typename T> 
__global__ void mulKernel2dRowWise(T* data, T* other, T* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other[j];
    }
}

__global__ void mulKernel2dColWise(double* data, double* other, double* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other[i];
    }
}

template <typename T>
__global__ void mulKernel2d(T* data, double other, T* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other;
    }
}

template<typename T> 
__global__ void dotKernel(T* data, T* other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] += data[i] * other[i];
    }
}


template <typename T>
__global__ void dotKernel2d(T *a, T *b, T *result, size_t widthA, size_t heightA, size_t widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        T sum = 0.0;
        for (int k = 0; k < widthA; ++k) {
            sum += a[row * widthA + k] * b[k * widthB + col];
        }
        result[row * widthB + col] = sum;
    }
}


template<typename T>
__global__ void dotKernel(T* data, double other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other;
    }
}


template <typename T>
__global__ void fillRandomKernel(T* data, int size, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        curandState_t state;
        curand_init(seed, i, 0, &state);
        data[i] = curand_uniform(&state);
    }
}

template <typename T>
__global__ void fillRandomKernel2d(T* data, size_t x, size_t y, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        curandState_t state;
        curand_init(seed, i * y + j, 0, &state);
        data[i * y + j] = curand_uniform(&state);
    }
}
