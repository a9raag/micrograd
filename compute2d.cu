#include <iostream> 
#include "include/compute2d.h"
// #include "include/cuda_compute.h"
// #include "cuda_compute.cu"
#include <stdexcept>
#include "compute2d.h"
using namespace std;

template <typename T>
Compute2D<T>::Compute2D(){
};

template <typename T>
Compute2D<T>::~Compute2D(){
    cudaFree(this->data);
}

template <typename T>
void Compute2D<T>:: setData(T* data){
    this->data = data;
}
template <typename T>
T* Compute2D<T>::getData(){
    return this->data;
}

template <typename T>
Compute2D<T>::Compute2D(int x, int y){
    this->data = new T[x * y];
    this->size = x * y;
    this->shape[0] = x;
    this->shape[1] = y;
    int allocSize = this->size * sizeof(T);

    this->block = dim3(x, y);
    this->grid = dim3((x + this->block.x - 1) / this->block.x, (y + this->block.y - 1) / this->block.y);
    
    if(cudaMallocManaged(&this->data, allocSize) != cudaSuccess){
        cout<<"Compute2D: Error in allocating memory"<<endl;
        // TODO: Add error message
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    cudaDeviceSynchronize();
    
}

template <typename T>
T* Compute2D<T>::add(BaseCompute<T>& compute){
    // if (size != this->size){
    //     throw invalid_argument("Size of the two arrays must be the same");
    // }
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T*  Compute2D<T>::add(double b){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel2d<<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T* Compute2D<T>::mul(BaseCompute<T>& compute){
    if (size != this->size){
        throw invalid_argument("Size of the two arrays must be the same");
    }
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel2d<T><<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T* Compute2D<T>::mul(double b){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T* Compute2D<T>::dot(BaseCompute<T>& compute){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    size_t heightA = shape[0];
    size_t widthA = shape[1];
    size_t widthB = compute.getShape()[1];
    // dotKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);
    matrixDotProduct<<<this->grid, this->block>>>(this->data, compute.getData(), result, widthA, heightA, widthB);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T* Compute2D<T>::pow(double n){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    powKernel2d<T><<<this->grid, this->block>>>(this->data, result, n, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
T* Compute2D<T>::tanh(){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    tanhKernel2d<T><<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
    cudaDeviceSynchronize();
    return result;
}

template <typename T>
void Compute2D<T>::fill(T value){
    fillKernel2d<T><<<this->grid, this->block>>>(this->data, value, shape[0], shape[1]);
    cudaDeviceSynchronize();
}

template <typename T>
void Compute2D<T>::fillRandom(unsigned int seed)
{
    fillRandomKernel2d<T><<<this->grid, this->block>>>(this->data, shape[0], shape[1], seed);
    cudaDeviceSynchronize();
}