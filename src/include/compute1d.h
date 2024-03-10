#pragma once
#ifndef COMPUTE1D_H
#define COMPUTE1D_H

#include "base_compute.h"
#include <vector>

template <typename T>
class Compute1D : public BaseCompute<T> {
private: 
    T* data;
    size_t size;
    int threadsPerBlock;
    int blocksPerGrid;
    int allocSize;
    size_t shape[1];

public:
    Compute1D();
    ~Compute1D();
    T* getData();
    void setData(T* data);
    size_t getSize(){
        return size;
    }
    size_t* getShape(){
        return this->shape;
    }
    Compute1D(size_t size);
    Compute1D(std::vector<T> hdata, size_t size);
    T* add(BaseCompute<T>& compute);
    T* add(double b);
    T* dot(BaseCompute<T>& compute);
    T* dot(double b);
    T* mul(BaseCompute<T>& compute);
    T* mul(double b);

    T* greater(BaseCompute<T>& compute);
    T* greater(double b);
    T* less(BaseCompute<T>& compute);
    T* less(double b);
    T* equal(BaseCompute<T>& compute);
    T* equal(double b);
    T* greaterEqual(BaseCompute<T>& compute);
    T* greaterEqual(double b);
    T* lessEqual(BaseCompute<T>& compute);
    T* lessEqual(double b);
    

    T* pow(double n);
    T* tanh();
    T* log();
    T* exp();
    
    
    T* sigmoid();
    T* relu();

    T* sum();
    T* sum(int axis);

    T* subArray(vector<vector<size_t>> dimRanges);

    void fill(T val);
    void fillRandom(unsigned int seed);
};

#endif // COMPUTE1D_H
