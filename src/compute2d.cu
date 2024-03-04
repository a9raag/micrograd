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
T*  Compute2D<T>::add(double b){
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
T* Compute2D<T>::mul(BaseCompute<T>& other){
    if (size != this->size){
        throw invalid_argument("Size of the two arrays must be the same");
    }
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
T* Compute2D<T>::mul(double b){
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
T *Compute2D<T>::greater(double b)
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
T *Compute2D<T>::less(double b)
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
T *Compute2D<T>::equal(double b)
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
T *Compute2D<T>::greaterEqual(double b)
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
T *Compute2D<T>::lessEqual(double b)
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
T* Compute2D<T>::pow(double n){
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
T *Compute2D<T>::sum()
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
void Compute2D<T>::fill(T value){
    fillKernel2d<T><<<this->grid, this->block>>>(this->data, value, shape[0], shape[1]);

}

template <typename T>
void Compute2D<T>::fillRandom(unsigned int seed)
{
    fillRandomKernel2d<T><<<this->grid, this->block>>>(this->data, shape[0], shape[1], seed);

}