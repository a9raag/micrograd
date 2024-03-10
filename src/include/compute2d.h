#pragma once
#ifndef Compute2D_H
#define COMPUTE2D_H

#include "base_compute.h"

template <typename T>
class Compute2D: public BaseCompute<T> {
private: 
    int allocSize;
    size_t size; 
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

    T* transpose();

    T* add(BaseCompute<T>& compute);
    T* add(double b);
    T* dot(BaseCompute<T>& compute);
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
    size_t getSize(){
        return size;
    }
    size_t* getShape() {
        return this->shape;
    }
};

#endif // COMPUTE2D_H
