#pragma once
#ifndef ENGINE_H
#define ENGINE_H

#include <iostream>
#include <functional>   
#include <set>
#include <unordered_set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include "../tensor.cu"
#include <vector>
#include <string>
using namespace std;

class Value: public std::enable_shared_from_this<Value>
{
private:
    Tensor<double> data;
    Tensor<double> grad;
    int size;
    set<shared_ptr<Value>> prev;
    std::string op;

public:
    function<void()> node_backward;
    std::string label;
    //default constructor
    Value();
    // Constructor
    // Value(const Tensor<double> data, std::initializer_list<Value> children, string op, string label);
    Value(Tensor<double> data, std::initializer_list<shared_ptr<Value>> children, string op, string label);
    // Value(const Tensor<T> &data, std::initializer_list<shared_ptr<Value<T>>> children, string op, string label);
    
    // Value(Tensor<T> &data, std::initializer_list<shared_ptr<Value<T>>> children, string op, string label);
    //copy constructor
    Value(const Value &other);
    //move constructor
    Value(Value &&other) noexcept;
    ~Value();
    shared_ptr<Value> pow(double n);

    Tensor<double> getData();
    void getData(Tensor<double> data);

    void setGrad(Tensor<double> grad);
    Tensor<double> getGrad();

    void set_grad_1();
    Value operator+(Value &other);
    bool operator<(const Value &other);
    shared_ptr<Value> tanh();
    shared_ptr<Value> neg();
    shared_ptr<Value> operator+(const shared_ptr<Value> &other);
    shared_ptr<Value> operator*(const shared_ptr<Value> &other);
    
    shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    shared_ptr<Value> operator/(const shared_ptr<Value> &other);

    // String representation
    friend std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v);
    void backward();
    

    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);  
    friend std::shared_ptr<Value> pow(const std::shared_ptr<Value>& lhs, const int n);
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
};

#endif