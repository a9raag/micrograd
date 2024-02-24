#pragma once
#ifndef BASE_COMPUTE_H
#define BASE_COMPUTE_H

#include <iostream>

template <typename T>
class BaseCompute {
public:
    virtual T* add(T* b, size_t* shape, size_t size) = 0;
    virtual T* add(double b) = 0;
    virtual T* mul(T* b, size_t* shape, size_t size) = 0;
    virtual T* mul(double b) = 0;
    virtual T* dot(T* b, size_t* shape, size_t size) = 0;
    // virtual T* dot(double b, size_t* shape, size_t size) = 0;
    virtual T* neg(){
        return this->mul(-1);
    
    };
    virtual T* pow(double n) = 0;
    virtual T* tanh() = 0;
    virtual void fill(T val) = 0;

    virtual void fillRandom(unsigned int seed) = 0;

    virtual T* getData() = 0;
    virtual void setData(T* data) = 0;
};

#endif // BASE_COMPUTE_H
