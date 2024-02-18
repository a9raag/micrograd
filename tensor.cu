#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <array> 
#include <memory>
#include <set> 
#include <functional>
#include "1dcompute.cu"

using namespace std;
template <typename T>
class Tensor {
        
private:
    size_t totalSize;
    unique_ptr<BaseCompute<T>> dataCompute; 
    unique_ptr<BaseCompute<T>> gradCompute;

    // Convert multidimensional indices to linear index
    size_t getLinearIndex(std::vector<int> coords) const {
        if (coords.size() != shape.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions.");
        }

        size_t linearIndex = 0; 
        size_t dimProduct = 1;
        for(int i = coords.size() - 1; i >= 0; --i) {
            linearIndex += coords[i] * dimProduct;
            dimProduct *= shape[i];
        }
        return (size_t)linearIndex;
    }
public:
    // declare complie-time constants 
    // value of following variables will be known at compile time
    size_t size;
    size_t ndims;
    vector<size_t> shape;

    Tensor(vector<size_t> shape) {
        this->ndims = shape.size();
        this->shape = shape;
        this->size = 1;
        for (size_t dim : shape) {
            if (dim == 0) throw std::invalid_argument("Dimension size cannot be zero.");
            size *= dim;
        }

        if (shape.empty()) throw std::invalid_argument("Dimensions cannot be empty.");

        totalSize = 1;
        for (size_t dim : shape) {
            if (dim == 0) throw std::invalid_argument("Dimension size cannot be zero.");
            totalSize *= dim;
        }

        dataCompute = make_unique<Compute1D<T>>(size);
        gradCompute  = make_unique<Compute1D<T>>(size);
    }

    //copy constructor
    Tensor(const Tensor& other) {
        this->ndims = other.ndims;
        this->shape = other.shape;
        this->size = other.size;
        this->totalSize = other.totalSize;
        this->dataCompute = make_unique<Compute1D<T>>(other.size);
        this->gradCompute = make_unique<Compute1D<T>>(other.size);
        this->dataCompute->setData(other.dataCompute->getData());
        this->gradCompute->setData(other.gradCompute->getData());
    }
    //default constructor
    Tensor() {
        this->ndims = 0;
        this->size = 0;
        this->totalSize = 0;
    }
    //move constructor
    Tensor(Tensor&& other) {
        this->ndims = other.ndims;
        this->shape = other.shape;
        this->size = other.size;
        this->totalSize = other.totalSize;
        this->dataCompute = move(other.dataCompute);
        this->gradCompute = move(other.gradCompute);
    }
    Tensor<T> operator=(const Tensor<T>& other) {
        this->ndims = other.ndims;
        this->shape = other.shape;
        this->size = other.size;
        this->totalSize = other.totalSize;
        this->dataCompute = make_unique<Compute1D<T>>(other.size);
        this->gradCompute = make_unique<Compute1D<T>>(other.size);
        this->dataCompute->setData(other.dataCompute->getData());
        this->gradCompute->setData(other.gradCompute->getData());
        return *this;
    }
    Tensor<T> operator=(Tensor<T>& other) {
        this->ndims = other.ndims;
        this->shape = other.shape;
        this->size = other.size;
        this->totalSize = other.totalSize;
        this->dataCompute = move(other.dataCompute);
        this->gradCompute = move(other.gradCompute);
        return *this;
    }
    template <typename ... Args>
    T& operator()(Args ... args) {
        auto idx = getLinearIndex({args...});
        return this->dataCompute->getData()[idx];
    }

    void setData(vector<T> data){
        this->dataCompute->setData(data);
        // this->data = data;
    }

    void setData(T* data){
        this->dataCompute->setData(data);
        // this->data = vector<T>(data, data + size);
    }
    vector<T> getData(){
        return this->dataCompute->getData();
    }
    void print_recursive(ostream& os , size_t i, size_t j) const{
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
    
    friend ostream &operator<<(ostream& os, const Tensor& t) {
        
        t.print_recursive(os, 0, 0);
        return os;
    }
    Tensor neg() {
        Tensor<T> result = Tensor<T>(shape);
        result.setData(dataCompute->neg());
        return result;
    }

    Tensor neg() const {
        Tensor<T> result = Tensor<T>(shape);
        result.setData(dataCompute->neg());
        return result;
    }

    Tensor pow(double n){
        Tensor<T> result = Tensor<T>(shape);
        result.setData(dataCompute->pow(n));
        return result;
    }

    Tensor tanh(){ 
        Tensor<T> result = Tensor<T>(shape);
        result.setData(dataCompute->tanh());
        return result;
    }

    void fill(T val){
        dataCompute->fill(val);
    }




    Tensor operator*(const Tensor &other){ 
        Tensor<T> result = Tensor<T>(shape);
        T* c = dataCompute->dot(other.dataCompute->getData());
        result.setData(c);
        return result;
    }

    Tensor operator*(const double other){ 
        Tensor<T> result = Tensor<T>(shape);
        T* c = dataCompute->dot(other);
        result.setData(c);
        return result;
    }

    Tensor operator/(Tensor &other){
        return *this * other.pow(-1);
    }
    
    
    Tensor operator+(const Tensor& other){

        Tensor<T> result = Tensor<T>(shape);
        T* c = dataCompute->add(other.dataCompute->getData());
        result.setData(c);
        return result;
    }
    Tensor operator+(const Tensor& other) const{

        Tensor<T> result = Tensor<T>(shape);
        T* c = dataCompute->add(other.dataCompute->getData());
        result.setData(c);
        return result;
    }
    friend Tensor operator+(const Tensor& t1, const Tensor& t2) {
        return t1.operator+(t2);
    }
    
    Tensor operator-(const Tensor &other){
        return *this + other.neg();
    }

    friend Tensor operator-(const Tensor& t1, const Tensor& t2) {
        return t1.operator-(t2);
    }

    friend Tensor operator*(const Tensor& t1, const Tensor& t2) {
        return t1 * t2;
    }
    friend Tensor operator/(const Tensor& t1, const Tensor& t2) {
        return t1 * t2.pow(-1);
    }   

};

// // Example usage
// int main() {
//     Tensor<int, 3, 3, 3> t;

    
//     // Assign random values to array
//     int value = 0;
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0;  j < 3; ++j) {
//             for (int k = 0; k < 3; ++k) {
//                 t(i, j, k) = value;
//                 value++;
//             }
//         }
//     }
//     cout << t << endl;
//     return 0;
// }
