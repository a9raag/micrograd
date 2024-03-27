#pragma once
#ifndef BASE_COMPUTE_H
#define BASE_COMPUTE_H

#include <iostream>
#include <vector> 
using namespace std;

template <typename T>
class BaseCompute {
    protected: 
        size_t* shape;
        
    public:
        ~BaseCompute() = default;
        virtual T* transpose() {
            throw std::runtime_error("Not implemented");
        }
        virtual T* add(BaseCompute<T>& compute) = 0;
        virtual T* add(float b) = 0;
        virtual T* mul(BaseCompute<T>& compute) = 0;
        virtual T* mul(float b) = 0;
        virtual T* dot(BaseCompute<T>& compute) = 0;
        virtual T* neg(){
            return this->mul(-1);
        
        };

        virtual T* greater(BaseCompute<T>& compute) = 0;
        virtual T* greater(float b) = 0;
        virtual T* less(BaseCompute<T>& compute) = 0;
        virtual T* less(float b) = 0;
        virtual T* equal(BaseCompute<T>& compute) = 0;
        virtual T* equal(float b) = 0;
        virtual T* greaterEqual(BaseCompute<T>& compute) = 0;
        virtual T* greaterEqual(float b) = 0;
        virtual T* lessEqual(BaseCompute<T>& compute) = 0;
        virtual T* lessEqual(float b) = 0;

        virtual T* pow(float n) = 0;
        virtual T* tanh() = 0;
        
        virtual T* sum() = 0;
        virtual T* sum(int axis) = 0;

        virtual T* subArray(vector<vector<size_t>> dimRanges) = 0;

        virtual T* log() = 0;
        virtual T* exp() = 0;

        virtual T* sigmoid() = 0;
        virtual T* relu() = 0;


        virtual void fill(T val) = 0;
        virtual void fillRandom(unsigned int seed) = 0;

        virtual T* getData() = 0;
        virtual void setData(T* data) = 0;
        virtual size_t getSize() = 0;
        virtual size_t* getShape() {
            return this->shape;
        }

        virtual int* toInt() = 0;
        virtual float* toFloat() = 0;

        virtual T* fancyIndexing(vector<vector<size_t>> indices) = 0;
};

#endif // BASE_COMPUTE_H
