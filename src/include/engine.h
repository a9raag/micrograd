#pragma once
#ifndef ENGINE_H
#define ENGINE_H

#include "tensor.h"
#include <functional>
#include <set>
#include <initializer_list>

using namespace std;

class Value: public std::enable_shared_from_this<Value>
{
private:
    Tensor<float> data;
    Tensor<float> grad;
    int size;
    set<shared_ptr<Value>> prev;
    std::string op;

public:
    function<void()> node_backward;
    std::string label;
    //default constructor
    Value();
    // Constructor with data
    template <typename T>
    Value(initializer_list<T> data);
    // Constructor
    Value(Tensor<float> data, std::initializer_list<shared_ptr<Value>> children, string op, string label);
    //copy constructor
    Value(const Value &other);
    //move constructor
    Value(Value &&other) noexcept;
    ~Value();
    shared_ptr<Value> pow(float n);

    Tensor<float> getData();
    void setData(Tensor<float> data);

    void setGrad(Tensor<float> grad);
    Tensor<float> getGrad();

    void set_grad_1();

    void zero_grad();

    shared_ptr<Value> subTensor(vector<vector<size_t>> dimRanges);
    

    shared_ptr<Value> sum();
    shared_ptr<Value> sum(int axis);

    shared_ptr<Value> mean();

    shared_ptr<Value> tanh();
    shared_ptr<Value> neg();
    
    shared_ptr<Value> exp();
    shared_ptr<Value> log();

    shared_ptr<Value> sigmoid();
    shared_ptr<Value> relu();



    shared_ptr<Value> operator+(const float &other);
    shared_ptr<Value> operator+(const shared_ptr<Value> &other);

    shared_ptr<Value> operator*(const float &other);
    shared_ptr<Value> operator*(const shared_ptr<Value> &other);

    shared_ptr<Value> operator-(const float &other);
    shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);

    shared_ptr<Value> operator/(const float &other);
    shared_ptr<Value> operator/(const shared_ptr<Value> &other);

    shared_ptr<Value> dot(const shared_ptr<Value> &other);

    // String representation
    friend std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v);
    void backward();
    
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const float& rhs);
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const float& rhs);
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);    

    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const float& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);  
    
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const float& rhs);
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);

};

#endif