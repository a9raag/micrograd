#pragma once
#ifndef COMPUTE1D_H
#define COMPUTE1D_H

#include "base_compute.h"
#include <vector>

template <typename T>
class Compute1D : public BaseCompute<T> {
private: 
    int threadsPerBlock;
    int blocksPerGrid;
    int allocSize;
    long size; 
    T* data;
    size_t shape[1];

public:
    Compute1D();
    ~Compute1D();
    T* getData();
    void setData(T* data);
    Compute1D(long size);
    Compute1D(std::vector<T> hdata, int size);
    T* add(BaseCompute<T>& compute);
    T* add(double b);
    T* dot(BaseCompute<T>& compute);
    T* dot(double b, size_t* shape, size_t size);
    T* mul(BaseCompute<T>& compute);
    T* mul(double b);
    T* pow(double n);
    T* tanh();
    void fill(T val);
    void fillRandom(unsigned int seed);
    size_t* getShape() {
        return this->shape;
    }
};

#endif // COMPUTE1D_H
