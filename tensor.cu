#include "include/tensor.h"
#include "include/compute1d.h"
#include "include/compute2d.h"
// #include <compute1d.h>
// #include <compute2d.h>

using namespace std;
// get Compute Type from shape
template <typename T>
unique_ptr<BaseCompute<T>> getComputeType(vector<size_t> shape) {
    if (shape.size() == 1) {
        return make_unique<Compute1D<T>>(shape[0]);
    } else if (shape.size() == 2) {
        return make_unique<Compute2D<T>>(shape[0], shape[1]);
    } else {
        throw std::invalid_argument("Only 1D and 2D tensors are supported.");
    }
}

template <typename T>
Tensor<T>::Tensor() {
    this->ndims = 0;
    this->size = 0;
    this->totalSize = 0;
}

template <typename T>
Tensor<T>::Tensor(vector<size_t> shape) {
    this->ndims = shape.size();
    this->shape = shape;
    this->size = 1;
    for (size_t dim : shape) {
        if (dim == 0) 
            throw std::invalid_argument("Dimension size cannot be zero.");
        size *= dim;
    }

    if (shape.empty()) throw std::invalid_argument("Dimensions cannot be empty.");

    totalSize = 1;
    for (size_t dim : shape) {
        if (dim == 0) throw std::invalid_argument("Dimension size cannot be zero.");
        totalSize *= dim;
    }
    dataCompute = getComputeType<T>(shape);
    gradCompute = getComputeType<T>(shape);
}

template <typename T>
Tensor<T>::Tensor(const Tensor& other) {
    this->ndims = other.ndims;
    this->shape = other.shape;
    this->size = other.size;
    this->totalSize = other.totalSize;
    this->dataCompute = getComputeType<T>(shape);
    this->gradCompute = getComputeType<T>(shape);
    this->dataCompute->setData(other.dataCompute->getData());
    this->gradCompute->setData(other.gradCompute->getData());
}



template <typename T>
Tensor<T>::Tensor(Tensor&& other) {
    this->ndims = other.ndims;
    this->shape = other.shape;
    this->size = other.size;
    this->totalSize = other.totalSize;
    this->dataCompute = move(other.dataCompute);
    this->gradCompute = move(other.gradCompute);
}

template <typename T>
Tensor<T> Tensor<T>::operator=(const Tensor<T>& other) {
    this->ndims = other.ndims;
    this->shape = other.shape;
    this->size = other.size;
    this->totalSize = other.totalSize;
    this->dataCompute = getComputeType<T>(shape);
    this->gradCompute = getComputeType<T>(shape);
    this->dataCompute->setData(other.dataCompute->getData());
    this->gradCompute->setData(other.gradCompute->getData());
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator=(Tensor<T>& other) {
    this->ndims = other.ndims;
    this->shape = other.shape;
    this->size = other.size;
    this->totalSize = other.totalSize;
    this->dataCompute = move(other.dataCompute);
    this->gradCompute = move(other.gradCompute);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::randomize() {
    return randomize(0);
}

template <typename T>
Tensor<T> Tensor<T>::randomize(unsigned int seed)
{
    dataCompute->fillRandom(seed);
    return *this;
}
template <typename T>
size_t Tensor<T>::getLinearIndex(std::vector<int> coords) const
{
    if (coords.size() != shape.size())
    {
        throw std::invalid_argument("Number of indices must match number of dimensions.");
    }

    size_t linearIndex = 0;
    size_t dimProduct = 1;
    for (int i = coords.size() - 1; i >= 0; --i)
    {
        linearIndex += coords[i] * dimProduct;
        dimProduct *= shape[i];
    }
    return (size_t)linearIndex;
}

template <typename T>
template <typename ... Args>
T& Tensor<T>::operator()(Args ... args) {
    auto idx = getLinearIndex({args...});
    return this->dataCompute->getData()[idx];
}

template <typename T>
void Tensor<T>::setData(vector<T> data){
    this->dataCompute->setData(data);
}

template <typename T>
void Tensor<T>::setData(T* data){
    this->dataCompute->setData(data);
}

template <typename T>
vector<T> Tensor<T>::getData(){
    return this->dataCompute->getData();
}

template <typename T>
void Tensor<T>::print_recursive(ostream& os , size_t i, size_t j) const{
    auto data = dataCompute->getData();
    
    if (i == ndims - 1){
        os << "[";
        for (int k = 0;  k < shape[i]; ++k) {
            os << data[j * shape[i] + k];
            if (k != shape[i] - 1) os << ", ";
        }
        os << "]";
        return;
    }
    os << "[";
    for (int k = 0;  k < shape[i]; ++k) {
        print_recursive(os, i + 1, j * shape[i] + k);
        if (k != shape[i] - 1){
            os << ", ";
            os<<endl;
        }
    }
    os << "]";
}

template <typename T>
ostream& operator<<(ostream& os, const Tensor<T>& t) {
    t.print_recursive(os, 0, 0);
    return os;
}

// template <typename T>
// Tensor<T> Tensor<T>::neg() {
//     Tensor<T> result = Tensor<T>(shape);
//     result.setData(dataCompute->neg());
//     return result;
// }

template <typename T>
Tensor<T> Tensor<T>::neg() const {
    Tensor<T> result = Tensor<T>(shape);
    result.setData(dataCompute->neg());
    return result;  
}

template <typename T>
Tensor<T> Tensor<T>::pow(double n){
    Tensor<T> result = Tensor<T>(shape);
    result.setData(dataCompute->pow(n));
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::tanh(){ 
    Tensor<T> result = Tensor<T>(shape);
    result.setData(dataCompute->tanh());
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::fill(T val){
    dataCompute->fill(val);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::dot(Tensor &other){
    if (ndims == 1 || other.ndims == 1){
        Tensor<T> result = Tensor<T>({1});
        T* c = dataCompute->dot(*other.dataCompute);
        result.setData(c);
        
        return result;

    }
    if (shape[1] != other.shape[0]){
        throw std::invalid_argument("Dot product is only defined for tensors with matching inner dimensions.");
    }
    vector<size_t> newShape = {shape[0], other.shape[1]};
    Tensor<T> result = Tensor<T>(newShape);
    vector<size_t> compute_shape = {shape[0], shape[1], other.shape[1]};
    T* c = dataCompute->dot(*other.dataCompute);
    result.setData(c);
    return result;
}


template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& other){
    if (other.size == 1 ){
        return *this + (double) other.dataCompute->getData()[0];
    }
    // if (shape != other.shape){
    //     throw std::invalid_argument("Tensors must have the same shape to be added.");
    // }
    Tensor<T> result = Tensor<T>(shape);
    T* c = dataCompute->add(*other.dataCompute);
    result.setData(c);
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const double other)
{
    Tensor<T> result = Tensor<T>(shape);
    T* c = dataCompute->add(other);
    result.setData(c);
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor &other){ 
    if (other.size == 1 ){
        return *this * (double) other.dataCompute->getData()[0];
    }
    Tensor<T> result = Tensor<T>(shape);
    T* c = dataCompute->mul(*other.dataCompute);
    result.setData(c);
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const double &other){ 
    Tensor<T> result = Tensor<T>(shape);
    T* c = dataCompute->mul(other);
    result.setData(c);
    return result;
}   



template <typename T>
Tensor<T> Tensor<T>::operator/(Tensor &other){
    return *this * other.pow(-1);
}



template <typename T>
Tensor<T> operator+(const Tensor<T>& t1, const Tensor<T>& t2) {
    return t1.operator+(t2);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor &other){
    return *this + other.neg();
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& t1, const Tensor<T>& t2) {
    return t1.operator-(t2);
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& t1, const Tensor<T>& t2) {
    return t1 * t2;
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& t1, const Tensor<T>& t2) {
    return t1 * t2.pow(-1);
}

template <typename T>
Tensor<T> operator*(const Tensor<T> &t1, const double& t2)
{
    return t1 * t2;
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &t1, const double& t2)
{
    return t1 + t2;
}

