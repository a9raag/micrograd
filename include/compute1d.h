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

public:
    Compute1D();
    ~Compute1D();
    T* getData();
    void setData(T* data);
    Compute1D(long size);
    Compute1D(std::vector<T> hdata, int size);
    T* add(T* b, size_t* shape, size_t size);
    T* add(double b);
    T* dot(T* b, size_t* shape, size_t size);
    T* dot(double b, size_t* shape, size_t size);
    T* mul(T* b, size_t* shape, size_t size);
    T* mul(double b);
    T* pow(double n);
    T* tanh();
    void fill(T val);
    void fillRandom(unsigned int seed);
};

#endif // COMPUTE1D_H
