#pragma once
#ifndef Compute2D_H
#define COMPUTE2D_H

#include "base_compute.h"

template <typename T>
class Compute2D: public BaseCompute<T> {
private: 
    int threadsPerBlock;
    int blocksPerGrid;
    int allocSize;
    long size; 
    dim3 block;
    dim3 grid;
    T* data;
    size_t shape[2];

public:
    Compute2D();
    ~Compute2D();
    T* getData();
    void setData(T* data);
    Compute2D(int x, int y);
    T* add(T* b, size_t* shape, size_t size);
    T* add(double b, size_t* shape, size_t size);
    T* dot(T* b, size_t* shape, size_t size);
    T* mul(T* b, size_t* shape, size_t size);
    T* mul(double b);
    T* pow(double n);
    T* tanh();
    void fill(T val);
};

#endif // COMPUTE2D_H
