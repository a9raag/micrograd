#pragma once
#ifndef CPU_COMPUTE1D_H
#define CPU_COMPUTE1D_H

#include "base_compute.h"
#include <vector>

template <typename T>
class CpuCompute1D : public BaseCompute<T> {
private:
    T* data;
    size_t size;
    size_t shape[1];

public:
    CpuCompute1D();
    ~CpuCompute1D();
    CpuCompute1D(size_t size);

    T* getData();
    void setData(T* src);
    size_t getSize() { return size; }
    size_t* getShape() { return this->shape; }

    T* add(BaseCompute<T>& compute);
    T* add(float b);
    T* dot(BaseCompute<T>& compute);
    T* mul(BaseCompute<T>& compute);
    T* mul(float b);

    T* greater(BaseCompute<T>& compute);
    T* greater(float b);
    T* less(BaseCompute<T>& compute);
    T* less(float b);
    T* equal(BaseCompute<T>& compute);
    T* equal(float b);
    T* greaterEqual(BaseCompute<T>& compute);
    T* greaterEqual(float b);
    T* lessEqual(BaseCompute<T>& compute);
    T* lessEqual(float b);

    T* pow(float n);
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

    int* toInt();
    float* toFloat();

    T* fancyIndexing(vector<vector<size_t>> indices);
};

#endif // CPU_COMPUTE1D_H
