#include <iostream> 
#include "include/compute1d.h"
#include "cuda_compute.cu"
#include <thrust/device_vector.h>

#include <stdexcept>
#include "compute1d.h"


using namespace std;

template <typename T>
Compute1D<T>::Compute1D() {
    this->size = 0;
    this->data = nullptr;
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
    cudaMemcpy(this->data, data, this->size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T> 
void Compute1D<T>::allocateMemory(T* data, size_t size) {
    cudaDeviceProp prop;
    int deviceId;
    cudaGetDevice(&deviceId); // Get current device ID
    cudaGetDeviceProperties(&prop, deviceId); // Get device properties
    this->threadsPerBlock = prop.maxThreadsPerBlock;
    // int blockSize = (maxThreadsPerBlock < 256) ? maxThreadsPerBlock : 256; // Example choice, considering hardware limit

    this->blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    this->size = size;
    if(cudaMallocManaged(&this->data, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    if (cudaMemcpy(this->data, data, size * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cout<<"Error in copying data to GPU"<<endl;
        cudaFree(this->data);
    }
}

template <typename T>
Compute1D<T>::Compute1D(size_t size)
{
    this->data = new T[size];
    this->shape[0] = size;
    this->size = size;
    this->threadsPerBlock = 32;
    this->blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize data on GPU
    int allocSize = this->size * sizeof(T);
    if(cudaMallocManaged(&this->data, allocSize) != cudaSuccess){
        cerr<<"Compute1D: Error in allocating memory"<<endl;
        cerr<<"Tried to allocate: "<<allocSize<<" bytes"<<endl;
        cerr<<"Size: "<<size<<endl;
        cerr<<"Threads per block: "<<threadsPerBlock<<endl;
        cerr<<"Blocks per grid: "<<blocksPerGrid<<endl;
        cerr<<cudaGetErrorString(cudaGetLastError())<<endl;
        cerr<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
        throw invalid_argument("Error in allocating memory");
    }

}

template <typename T>
Compute1D<T>::Compute1D(vector<T> hdata, size_t dataSize) {
    *this = Compute1D(dataSize);
    if (cudaMemcpy(this->data, hdata.data(), allocSize, cudaMemcpyHostToDevice) != cudaSuccess) {
        cout<<"Error in copying data to GPU"<<endl;
        cudaFree(this->data);
    }
}

template <typename T>
T* Compute1D<T>::add(BaseCompute<T>& compute) {
    T* c;  
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);

    return c;
}

template <typename T>
T* Compute1D<T>::add(float b) {
    T* c;  
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);

    return c;
}

template <typename T>
T* Compute1D<T>::dot(BaseCompute<T>& compute){ 
    T *c; 
    
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"1dcompute:dot: Error in allocating memory"<<endl;
        throw runtime_error("1dcompute:dot Error in allocating memory");
    }
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, size);
    thrust::device_vector<T> d_vec(c, c + size);
    T sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<float>());
    
    T* out = new T[1];
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"1dcompute:dot: Error in allocating memory"<<endl;
        throw runtime_error("1dcompute:dot Error in allocating memory");
    }
    out[0] = sum;

    

    cudaFree(c);

    // dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
    // dotKernel2d<<<gridDim, blockDim>>>(this->data, b, c, shape[0], shape[1]);
    
    return out;

}

//TODO: refactor remove unused parameters
template <typename T>
T* Compute1D<T>::dot(float b){ 
    T *c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);

    return c;

}

template <typename T>
T* Compute1D<T>::mul(BaseCompute<T>& compute){ 

    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);

    return c;
}

template <typename T>
T* Compute1D<T>::mul(float b){ 
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    mulKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);

    return c;
}

template <typename T>
T *Compute1D<T>::greater(BaseCompute<T> &compute)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    greaterKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::greater(float b)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    greaterKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::greaterEqual(BaseCompute<T> &compute)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    greaterEqualKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::greaterEqual(float b)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    greaterEqualKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::less(BaseCompute<T> &compute)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    lessKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::less(float b)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    lessKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::lessEqual(BaseCompute<T> &compute)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    lessEqualKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::lessEqual(float b)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    lessEqualKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::equal(BaseCompute<T> &compute)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    equalKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, compute.getData(), c, this->size);
    return c;
}

template <typename T>
T *Compute1D<T>::equal(float b)
{
    T* c; 
    if(cudaMallocManaged(&c, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    equalKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, this->size);
    return c;
}

template <typename T>
T* Compute1D<T>::pow(float n){ 
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    powKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, n, this->size);
    return out;
}

template <typename T>
T* Compute1D<T>::tanh(){ 
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, this->size);


    return out;
}

template <typename T>
T *Compute1D<T>::log()
{
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    logKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, this->size);
    
    return out;
}

template <typename T>
T *Compute1D<T>::exp()
{
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    expKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, this->size);
    
    return out;

}

template <typename T>
T *Compute1D<T>::sigmoid()
{
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    sigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, this->size);
    
    return out;
}

template <typename T>
T *Compute1D<T>::relu()
{
    T* out; 
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, this->size);
    return out;
}

template <typename T>
T *Compute1D<T>::sum()
{
    thrust::device_vector<T> d_vec(data, data + this->size);
    T sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<T>());
    T* out = new T[1];
    if(cudaMallocManaged(&out, this->size * sizeof(T)) != cudaSuccess){
        cout<<"1dcompute:dot: Error in allocating memory"<<endl;
        throw runtime_error("1dcompute:dot Error in allocating memory");
    }
    out[0] = sum;
    
    return out;
}

template <typename T>
T* Compute1D<T>::sum(int axis)
{
    if(axis != 0){
        throw invalid_argument("Invalid axis for 1D array. Only axis=0 is allowed.");
    }
    return sum();
}
template <typename T>
T *Compute1D<T>::subArray(vector<vector<size_t>> dimRange)
{
    if(dimRange.size() != 1){
        throw invalid_argument("Invalid start index for 1D array. Only 1D start index is allowed.");
    }
    if (dimRange[0].size() != 2)
    {
        throw invalid_argument("Start and end index must be provided for 1D array.");
    }
    T* out;
    size_t newSize = dimRange[0][1] - dimRange[0][0];
    if(cudaMallocManaged(&out, newSize * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        throw runtime_error("Error in allocating memory");
    }
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((newSize + threadsPerBlock.x - 1) / threadsPerBlock.x);
    subArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, dimRange[0][0], newSize);
    return out;
}
template <typename T>
void Compute1D<T>::fill(T val)
{
    fillKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, val, size);

}
template <typename T>
void Compute1D<T>::fillRandom(unsigned int seed){
    fillRandomKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, size, seed);

}