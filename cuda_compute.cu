#include <cuda_runtime.h>
#include <vector> 
#include <math.h>
template <typename T>
__global__ void fillKernel(T* data, T val, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = val;
    }
}

__global__ void set_ones2d(double** grad, int *size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size[0] && j < size[1]) {
        grad[i][j] = 1.0;
    }
}

template <typename T>
__global__ void powKernel(T* a, T *out, double n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = pow(a[i], n);
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

__global__ void addGrad(double* grad, double* outGrad, double* otherGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += 1.0 * outGrad[i];
        otherGrad[i] += 1.0 * outGrad[i];
    }
}

template<typename T>
__global__ void dotKernel(T* data, T* other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other[i];
    }
}


template<typename T>
__global__ void dotKernel(T* data, double other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other;
    }
}


__global__ void dotGrad(double* grad, double * data, double* outGrad, double* otherData, double* otherGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += otherData[i] * outGrad[i];
        otherGrad[i] += data[i] * outGrad[i];
    }
}