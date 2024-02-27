#pragma once
#ifndef Compute2D_H
#define COMPUTE2D_H

#include "base_compute.h"

template <typename T>
class Compute2D: public BaseCompute<T> {
private: 
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
    T* add(BaseCompute<T>& compute);
    T* add(double b);
    T* dot(BaseCompute<T>& compute);
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

#endif // COMPUTE2D_H
