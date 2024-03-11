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
__global__ void subArrayKernel(T* data, T* result, size_t datax, size_t start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + start < datax) {
        result[i] = data[i + start];
    }
}

template <typename T> 
__global__ void subArrayKernel2d(T* data, T* result, size_t datax, size_t datay, size_t resultx, size_t resulty, size_t startx, size_t starty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i + startx < datax && j + starty < datay) {
        result[i * resulty + j] = data[(i + startx) * datay + j + starty];
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
__global__ void logKernel(T* data, T* out, size_t size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = logf(data[i]);
    }
}

template <typename T>
__global__ void logKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = logf(data[i * y + j]);
    }
}


template <typename T>
__global__ void expKernel(T* data, T* out, size_t size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = expf(data[i]);
    }
}

template <typename T>
__global__ void expKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = exp(data[i * y + j]);
    }
}



template <typename T>
__global__ void powKernel(T* a, T *out, float n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = pow(a[i], n);
    }
}

template <typename T> 
__global__ void powKernel2d(T* a, T* out, float n, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = pow(a[i * y + j], n);
    }
}

__global__ void cudaPowGrad(float* data, float *outGrad, float* grad, float n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += n * pow(data[i], n - 1) * outGrad[i];
    }
}

template <typename T>
__global__ void tanhKernel(T* data, T* out, int size){ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size){
        float x = data[i]; 
        out[i] = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    }
}

template <typename T> 
__global__ void tanhKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y){
        float x = data[i * y + j]; 
        out[i * y + j] = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    }    
}

__global__ void gradTanh(float* grad, float* outGrad, float* data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float t = (exp(2.0 * data[i]) - 1.0) / (exp(2.0 * data[i]) + 1.0);
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

template <typename T>
__global__ void greaterKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] > b[i];
    }
}

template <typename T>
__global__ void greaterKernel(T* a, float b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] > b;
    }
}

template <typename T>
__global__ void greaterKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] > b[i * y + j];
    }
}

template <typename T>
__global__ void greaterKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] > b;
    }
}

template <typename T>
__global__ void lessKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] < b[i];
    }
}

template <typename T> 
__global__ void lessKernel(T* a, float b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] < b;
    }
}

template <typename T>
__global__ void lessKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] < b[i * y + j];
    }
}

template <typename T>
__global__ void lessKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] < b;
    }
}

template <typename T>
__global__ void equalKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] == b[i];
    }
}


template <typename T>
__global__ void equalKernel(T* a, float b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] == b;
    }
}

template <typename T>
__global__ void equalKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] == b;
    }
}

template <typename T>
__global__ void equalKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] == b[i * y + j];
    }
}

template <typename T>
__global__ void greaterEqualKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] >= b[i];
    }
}

template <typename T>
__global__ void greaterEqualKernel(T* a, float b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] >= b;
    }
}

template <typename T>
__global__ void greaterEqualKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] >= b[i * y + j];
    }
}

template <typename T>
__global__ void greaterEqualKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] >= b;
    }
}

template <typename T>
__global__ void lessEqualKernel(T* a, T* b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] <= b[i];
    }
}

template <typename T>
__global__ void lessEqualKernel(T* a, float b, T* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] <= b;
    }
}

template <typename T>
__global__ void lessEqualKernel2d(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] <= b[i * y + j];
    }
}

template <typename T>
__global__ void lessEqualKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] <= b;
    }
}


template <typename T>
__global__ void addKernel(T* a, float b, T* c, int size) {
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
__global__ void addKernel2d(T* a, float b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b;
    }
}

template <typename T> 
__global__ void addKernel2dSingleton(T* a, T* b, T* c, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        c[i * y + j] = a[i * y + j] + b[0];
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

template <typename T>
__global__ void sumKernel2daxis0(T* a, T* c, size_t x, size_t y){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < y) {
        T sum = 0;
        for (int i = 0; i < x; i++) {
            sum += a[i * y + j];
        }
        c[j] = sum;
    } 
}

template <typename T>
__global__ void sumKernel2daxis1(T* a, T* c, size_t x, size_t y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < x) {
        T sum = 0;
        for (int j = 0; j < y; j++) {
            sum += a[i * y + j];
        }
        c[i] = sum;
    } 
}

__global__ void addGrad(float* grad, float* outGrad, float* otherGrad, int size) {
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
__global__ void mulKernel(T* data, float other, T* result, int size) {
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
__global__ void mulKernel2dSingleton(T* data, T* other, T* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other[0];
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

__global__ void mulKernel2dColWise(float* data, float* other, float* result, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        result[i * y + j] = data[i * y + j] * other[i];
    }
}

template <typename T>
__global__ void mulKernel2d(T* data, float other, T* result, size_t x, size_t y) {
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
__global__ void dotKernel(T* data, float other, T* result, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = data[i] * other;
    }
}

// uniform values between -1 and 1
template <typename T>
__global__ void fillRandomKernel(T* data, int size, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        curandState_t state;
        curand_init(seed, i, 0, &state);
        data[i] = 2 * curand_uniform(&state) - 1;
    }
}

template <typename T>
__global__ void fillRandomKernel2d(T* data, size_t x, size_t y, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        curandState_t state;
        curand_init(seed, i * y + j, 0, &state);
        data[i * y + j] = curand_uniform(&state) * 2 - 1;
    }
}

template <typename T>
__global__ void sigmoidKernel(T* data, T* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = 1.0 / (1.0 + expf(-data[i]));
    }
}

template <typename T>
__global__ void sigmoidKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = 1.0 / (1.0 + expf(-data[i * y + j]));
    }
}


template <typename T>
__global__ void gradSigmoid(float* grad, float* outGrad, float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float s = 1.0 / (1.0 + expf(-data[i]));
        grad[i] += s * (1 - s) * outGrad[i];
    }
}

template <typename T>
__global__ void reluKernel(T* data, T* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = data[i] > 0 ? data[i] : 0;
    }
}

template <typename T>
__global__ void reluKernel2d(T* data, T* out, size_t x, size_t y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < x && j < y) {
        out[i * y + j] = data[i * y + j] > 0 ? data[i * y + j] : 0;
    }
}

template <typename T>
__global__ void gradRelu(float* grad, float* outGrad, float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += data[i] > 0 ? outGrad[i] : 0;
    }
}