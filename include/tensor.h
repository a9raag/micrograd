#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include "base_compute.h"
#include <vector>

using namespace std;

template <typename T>
class Tensor {
private:
    size_t totalSize;
    unique_ptr<BaseCompute<T>> dataCompute; 
    unique_ptr<BaseCompute<T>> gradCompute;

    // Convert multidimensional indices to linear index
    

public:
    size_t size;
    size_t ndims;
    vector<size_t> shape;
    Tensor();
    Tensor(vector<size_t> shape);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    
    Tensor<T> operator=(const Tensor<T>& other);
    Tensor<T> operator=(Tensor<T>& other);
    
    size_t getLinearIndex(std::vector<int> coords) const;
    template <typename ... Args>
    T& operator()(Args ... args);

    void setData(vector<T> data);
    void setData(T* data);
    vector<T> getData();
    void print_recursive(ostream& os , size_t i, size_t j) const;

    Tensor neg();
    Tensor neg() const;
    Tensor pow(double n);
    Tensor tanh();
    void fill(T val);

    Tensor operator*(const Tensor &other);
    Tensor operator*(const double other);
    Tensor dot(Tensor &other);
    Tensor operator/(Tensor &other);
    Tensor operator+(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor &other);

    template <typename U>
    friend ostream &operator<<(ostream& os, const Tensor<U>& t);

    template <typename U>
    friend Tensor operator+(const Tensor<U>& t1, const Tensor<U>& t2);

    template <typename U>
    friend Tensor operator-(const Tensor<U>& t1, const Tensor<U>& t2);

    template <typename U>
    friend Tensor operator*(const Tensor<U>& t1, const Tensor<U>& t2);

    template <typename U>
    friend Tensor operator/(const Tensor<U>& t1, const Tensor<U>& t2);
};

#endif // TENSOR_H
