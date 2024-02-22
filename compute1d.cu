#include <iostream> 
#include "include/compute1d.h"
#include "cuda_compute.cu"
#include <stdexcept>
using namespace std;

template <typename T>
Compute1D<T>::Compute1D() {
}

template <typename T>
Compute1D<T>::~Compute1D() {
    cudaFree(this->data);
}

template <typename T>
T* Compute1D<T>::getData() {
    return this->data;
}

template <typename T>
void Compute1D<T>::setData(T* data) {
    this->data = data;
}

template <typename T>
Compute1D<T>::Compute1D(long size){
    data = new T[size];
    this->size = size;
    this->threadsPerBlock = 256;
    this->blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize data on GPU
    int allocSize = size * sizeof(T);
    if(cudaMallocManaged(&this->data, allocSize) != cudaSuccess){
        cout<<"Compute1D: Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw invalid_argument("Error in allocating memory");
    }
    cudaDeviceSynchronize();
}

template <typename T>
Compute1D<T>::Compute1D(vector<T> hdata, int size) {
    if (cudaMemcpy(this->data, hdata.data(), allocSize, cudaMemcpyHostToDevice) != cudaSuccess) {
        cout<<"Error in copying data to GPU"<<endl;
        cudaFree(this->data);
    }
    *this = Compute1D(size);
}

template <typename T>
T* Compute1D<T>::add(T* b, size_t* shape, size_t size) {
    T* c;  
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    cudaDeviceSynchronize();
    return c;
}

template <typename T>
T* Compute1D<T>::add(double b, size_t* shape, size_t size) {
    T* c;  
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    cudaDeviceSynchronize();
    return c;
}

template <typename T>
T* Compute1D<T>::dot(T* b, size_t* shape, size_t size){ 
    dim3 blockDim(16, 16);
    dim3 gridDim((shape[0] + blockDim.x - 1) / blockDim.x, (shape[1] + blockDim.y - 1) / blockDim.y);
    T *c; 
    
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"1dcompute:dot: Error in allocating memory"<<endl;
        throw runtime_error("1dcompute:dot Error in allocating memory");
    }
    dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    // dotKernel2d<<<gridDim, blockDim>>>(this->data, b, c, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return c;

}

template <typename T>
T* Compute1D<T>::dot(double b, size_t* shape, size_t size){ 
    T *c; 
    
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    cudaDeviceSynchronize();
    return c;

}

template <typename T>
T* Compute1D<T>::mul(T* b, size_t* shape, size_t size){ 
    T* c; 
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    cudaDeviceSynchronize();
    return c;
}

template <typename T>
T* Compute1D<T>::mul(double b){ 
    T* c; 
    if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    cudaDeviceSynchronize();
    return c;
}

template <typename T>
T* Compute1D<T>::pow(double n){ 
    T* out; 
    if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, n, size);
    cudaDeviceSynchronize();
    return out;
}

template <typename T>
T* Compute1D<T>::tanh(){ 
    T* out; 
    if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, size);

    cudaDeviceSynchronize();
    return out;
}

template <typename T>
void Compute1D<T>::fill(T val){
    fillKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, val, size);
    cudaDeviceSynchronize();
}
template <typename T>
void Compute1D<T>::fillRandom(unsigned int seed){
    fillRandomKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, size, seed);
    cudaDeviceSynchronize();
}