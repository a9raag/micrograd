#pragma once
#ifndef BASE_COMPUTE_H
#define BASE_COMPUTE_H

#include <iostream>

template <typename T>
class BaseCompute {
    private: 
        T* data;
        size_t* shape;
        size_t size;
    public:
        virtual T* add(BaseCompute<T>& compute) = 0;
        virtual T* add(double b) = 0;
        virtual T* mul(BaseCompute<T>& compute) = 0;
        virtual T* mul(double b) = 0;
        virtual T* dot(BaseCompute<T>& compute) = 0;
        virtual T* neg(){
            return this->mul(-1);
        
        };
        virtual T* pow(double n) = 0;
        virtual T* tanh() = 0;
        virtual void fill(T val) = 0;

    virtual void fillRandom(unsigned int seed) = 0;

    virtual T* getData() = 0;
    virtual void setData(T* data) = 0;
    virtual size_t* getShape() {
        return this->shape;
    }
};

#endif // BASE_COMPUTE_H
