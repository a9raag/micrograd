#include <iostream> 
#include <sstream>
#include "include/compute2d.h"
#include "include/compute1d.h"
// #include "include/cuda_compute.h"
// #include "cuda_compute.cu"
#include <stdexcept>
#include "compute2d.h"
using namespace std;

enum MATRIX_TYPE{
    ROW_WISE,
    COL_WISE,
    NORMAL, 
    SINGLETON
};

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
    cudaMemcpy(this->data, data, this->allocSize, cudaMemcpyHostToDevice);
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
    this->allocSize = x * y * sizeof(T);

    this->block = dim3(32, 32);
    this->grid = dim3((x + this->block.x - 1) / this->block.x, (y + this->block.y - 1) / this->block.y);
    
    if(cudaMallocManaged(&this->data, allocSize) != cudaSuccess){
        cerr<<"Error in allocating memory"<<endl;
        cerr<<"Block size is "<<this->block.x<<"x"<<this->block.y<<endl;
        cerr<<"Grid size is "<<this->grid.x<<"x"<<this->grid.y<<endl;
        cerr<<"Size of the array is "<<x<<"x"<<y<<endl;
        cerr<<"Tried to allocate "<<allocSize<<" bytes"<<endl;
        cerr<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }

    
}

template <typename T>
T* Compute2D<T>::transpose(){
    T* result = new T[shape[0] * shape[1]];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    transposeKernel2d<T><<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);

    return result;
}

template <typename T>
MATRIX_TYPE matrixType(BaseCompute<T>& lhs, BaseCompute<T>& rhs){
    if(rhs.getSize() == 1)  return SINGLETON;
    
    if(rhs.getShape()[0] == 1 && rhs.getShape()[1] == lhs.getShape()[1]) 
        return ROW_WISE;
    
    if(rhs.getShape()[0] == lhs.getShape()[0] && rhs.getShape()[1] == 1)
        return COL_WISE;
    
    if(typeid(rhs) == typeid(Compute1D<T>)  && lhs.getShape()[1] == rhs.getSize())
        return ROW_WISE;

    if (rhs.getShape()[0] == lhs.getShape()[0] && rhs.getShape()[1] == lhs.getShape()[1])
        return NORMAL;
    
    
    ostringstream error;
    error << "Shape of the two arrays must be the same. Shape of the first array is " << lhs.getShape()[0] << "x" << lhs.getShape()[1] << " and the shape of the second array is " << rhs.getShape()[0] << "x" << rhs.getShape()[1];
    throw invalid_argument(error.str());

}
template <typename T>
T* Compute2D<T>::add(BaseCompute<T>& other){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    MATRIX_TYPE type = matrixType(*this, other);
    switch (type)
    {
    case SINGLETON:
        addKernel2dSingleton<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case ROW_WISE:
        addKernel2dRowWise<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case COL_WISE:
        addKernel2dColWise<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case NORMAL:
        addKernel2d<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    default:
        break;
    }
    return result;
}

template <typename T>
T*  Compute2D<T>::add(float b){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    addKernel2d<<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T* Compute2D<T>::sum()
{
    thrust::device_vector<T> d_vec(data, data + size);
    T sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<T>());
    T* out = new T[1];
    if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
        cout<<"2dcompute:dot: Error in allocating memory"<<endl;
        throw runtime_error("2dcompute:dot Error in allocating memory");
    }
    out[0] = sum;
 
    return out;
}

template <typename T>
T* Compute2D<T>::sum(int axis)
{
    if (axis == 0)
    {
        T *result = new T[shape[1]];
        if (cudaMallocManaged(&result, shape[1] * sizeof(T)) != cudaSuccess)
        {
            cout << "Error in allocating memory" << endl;
            cout << cudaGetErrorString(cudaGetLastError()) << endl;
            throw runtime_error("Error in allocating memory");
        }
        sumKernel2daxis0<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
        return result;
    }
    else if (axis == 1)
    {
        T *result = new T[shape[0]];
        if (cudaMallocManaged(&result, shape[0] * sizeof(T)) != cudaSuccess)
        {
            cout << "Error in allocating memory" << endl;
            cout << cudaGetErrorString(cudaGetLastError()) << endl;
            throw runtime_error("Error in allocating memory");
        }
        sumKernel2daxis0<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
        return result;
    }
    else
    {
        throw invalid_argument("Axis must be 0 or 1");
    }
}


template <typename T>
T *Compute2D<T>::subArray(vector<vector<size_t>> dimRanges)
{
    if (dimRanges.size() != 2)
    {
        throw invalid_argument("dimRanges must be a 2D array");
    }
    if (dimRanges[0].size() != 2 || dimRanges[1].size() != 2)
    {
        throw invalid_argument("Each dimension range must be a 1D array of size 2");
    }
    for (auto range: dimRanges){
        if (range[1] > shape[0]){
            throw invalid_argument("Range must be within the shape of the array");
        }
    }
    size_t result_x = dimRanges[0][1] - dimRanges[0][0];
    size_t result_y = dimRanges[1][1] - dimRanges[1][0];
    T *result = new T[result_x * result_y];
    if (cudaMallocManaged(&result, result_x * result_x * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    dim3 tempBlock(result_x, result_y);
    dim3 tempGrid((result_x + tempBlock.x - 1) / tempBlock.x, (result_y + tempBlock.y - 1) / tempBlock.y);
    subArrayKernel2d<<<tempGrid, tempBlock>>>(this->data, result, shape[0], shape[1], result_x, result_y, dimRanges[0][0], dimRanges[1][0]);
    return result;
}

template <typename T>
T* Compute2D<T>::mul(BaseCompute<T>& other){
    if (size != this->size){
        throw invalid_argument("Size of the two arrays must be the same");
    }
    T* result;
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cerr<<"Error in allocating memory"<<endl;
        cerr<<cudaGetErrorString(cudaGetLastError())<<endl;
        cerr<<"Size of the array is "<<shape[0]<<"x"<<shape[1]<<endl;
        cerr<<"Tried to allocate "<<size * sizeof(T)<<" bytes"<<endl;
        throw runtime_error("Error in allocating memory");
    }

    MATRIX_TYPE type = matrixType(*this, other);
    switch (type)
    {
    case SINGLETON:
        mulKernel2dSingleton<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case ROW_WISE:
        mulKernel2dRowWise<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case COL_WISE:
        mulKernel2dColWise<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;
    case NORMAL:
        mulKernel2d<<<this->grid, this->block>>>(this->data, other.getData(), result, shape[0], shape[1]);
        break;

    default:
        break;
    }

    return result;
}

template <typename T>
T* Compute2D<T>::mul(float b){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }

    mulKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::greater(BaseCompute<T> &compute)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    if (shape[0] != compute.getShape()[0] || shape[1] != compute.getShape()[1])
    {
        ostringstream error;
        error << "Shape of the two arrays must be the same. Shape of the first array is " << shape[0] << "x" << shape[1] << " and the shape of the second array is " << compute.getShape()[0] << "x" << compute.getShape()[1];
        throw invalid_argument(error.str());
    }

    greaterKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::greater(float b)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    greaterKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::less(BaseCompute<T> &compute)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    if (shape[0] != compute.getShape()[0] || shape[1] != compute.getShape()[1])
    {
        ostringstream error;
        error << "Shape of the two arrays must be the same. Shape of the first array is " << shape[0] << "x" << shape[1] << " and the shape of the second array is " << compute.getShape()[0] << "x" << compute.getShape()[1];
        throw invalid_argument(error.str());
    }

    lessKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);

    return result;
}


template <typename T>
T *Compute2D<T>::less(float b)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    lessKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::equal(BaseCompute<T> &compute)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    if (shape[0] != compute.getShape()[0] || shape[1] != compute.getShape()[1])
    {
        ostringstream error;
        error << "Shape of the two arrays must be the same. Shape of the first array is " << shape[0] << "x" << shape[1] << " and the shape of the second array is " << compute.getShape()[0] << "x" << compute.getShape()[1];
        throw invalid_argument(error.str());
    }

    equalKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::equal(float b)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    equalKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::greaterEqual(BaseCompute<T> &compute)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    if (shape[0] != compute.getShape()[0] || shape[1] != compute.getShape()[1])
    {
        ostringstream error;
        error << "Shape of the two arrays must be the same. Shape of the first array is " << shape[0] << "x" << shape[1] << " and the shape of the second array is " << compute.getShape()[0] << "x" << compute.getShape()[1];
        throw invalid_argument(error.str());
    }

    greaterEqualKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::greaterEqual(float b)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    greaterEqualKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

    return result;
}

template <typename T>  
T *Compute2D<T>::lessEqual(BaseCompute<T> &compute)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    if (shape[0] != compute.getShape()[0] || shape[1] != compute.getShape()[1])
    {
        ostringstream error;
        error << "Shape of the two arrays must be the same. Shape of the first array is " << shape[0] << "x" << shape[1] << " and the shape of the second array is " << compute.getShape()[0] << "x" << compute.getShape()[1];
        throw invalid_argument(error.str());
    }

    lessEqualKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::lessEqual(float b)
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    lessEqualKernel2d<T><<<this->grid, this->block>>>(this->data, b, result, shape[0], shape[1]);

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

    if (shape[1] != compute.getShape()[0]){
        ostringstream error;
        error << "Width of the first array " <<shape[1]<< " must be equal to the height of the second array " << compute.getShape()[0];
        throw invalid_argument(error.str());
    }
    
    size_t heightA = shape[0];
    size_t widthA = shape[1];
    size_t widthB = compute.getShape()[1];


    dotKernel2d<<<this->grid, this->block>>>(this->data, compute.getData(), result, widthA, heightA, widthB);

    return result;
}

template <typename T>
T* Compute2D<T>::pow(float n){
    T* result = new T[size];
    if(cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess){
        cout<<"Error in allocating memory"<<endl;
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        throw runtime_error("Error in allocating memory");
    }
    powKernel2d<T><<<this->grid, this->block>>>(this->data, result, n, shape[0], shape[1]);

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

    return result;
}

template <typename T>
T *Compute2D<T>::log()
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    logKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
    return result;
}

template <typename T>
T *Compute2D<T>::exp()
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    expKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
    return result;

}

template <typename T>
T *Compute2D<T>::sigmoid()
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    sigmoidKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
    return result;
}

template <typename T>
T *Compute2D<T>::relu()
{
    T *result = new T[size];
    if (cudaMallocManaged(&result, size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    reluKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);
    return result;
}



template <typename T>
void Compute2D<T>::fill(T value){
    fillKernel2d<<<this->grid, this->block>>>(this->data, value, shape[0], shape[1]);

}

template <typename T>
void Compute2D<T>::fillRandom(unsigned int seed)
{
    fillRandomKernel2d<<<this->grid, this->block>>>(this->data, shape[0], shape[1], seed);

}

template <typename T>
int *Compute2D<T>::toInt()
{
    int *result = new int[size];
    if (cudaMallocManaged(&result, size * sizeof(int)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    toIntKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);

    return result;
}

template <typename T>
float *Compute2D<T>::toFloat()
{
    float *result = new float[size];
    if (cudaMallocManaged(&result, size * sizeof(float)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    toFloatKernel2d<<<this->grid, this->block>>>(this->data, result, shape[0], shape[1]);

    return result;
}

template <typename T>
T *Compute2D<T>::fancyIndexing(vector<vector<size_t>> indices)
{
    if (indices.size() != 2)
    {
        throw invalid_argument("Indices must be a 2D array");
    }
    if (indices[0].size() != 1 || indices[1].size() != 1)
    {
        throw invalid_argument("Each dimension range must be a 1D array of size 1");
    }
    if(indices[0].size() != indices[1].size()){
        throw invalid_argument("Each dimension range must be of the same size");
    }
    for (auto index: indices){
        if (index[0] > shape[0]){
            throw invalid_argument("Index must be within the shape of the array");
        }
    }
    size_t result_size = indices[0].size();
    T *result = new T[result_size];
    if (cudaMallocManaged(&result, result_size * sizeof(T)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    size_t* deviceIndicesX; 
    size_t* deviceIndicesY;
    if (cudaMallocManaged(&deviceIndicesX, result_size * sizeof(size_t)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }
    if (cudaMallocManaged(&deviceIndicesY, result_size * sizeof(size_t)) != cudaSuccess)
    {
        cout << "Error in allocating memory" << endl;
        cout << cudaGetErrorString(cudaGetLastError()) << endl;
        throw runtime_error("Error in allocating memory");
    }

    cudaMemcpy(deviceIndicesX, indices[0].data(), result_size * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndicesY, indices[1].data(), result_size * sizeof(size_t), cudaMemcpyHostToDevice);

    dim3 tempBlock(result_size, 1);
    dim3 tempGrid((result_size + tempBlock.x - 1) / tempBlock.x, 1);
    fancyIndexingKernel2d<<<tempGrid, tempBlock>>>(this->data, result, shape[0], shape[1], result_size, result_size, deviceIndicesX, deviceIndicesY);
    return result;
}
