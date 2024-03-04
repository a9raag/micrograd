#pragma once
#ifndef ENGINE_H
#define ENGINE_H

#include "tensor.h"
#include <functional>
#include <set>

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
    Value(Tensor<double> data, std::initializer_list<shared_ptr<Value>> children, string op, string label);
    //copy constructor
    Value(const Value &other);
    //move constructor
    Value(Value &&other) noexcept;
    ~Value();
    shared_ptr<Value> pow(double n);

    Tensor<double> getData();
    void setData(Tensor<double> data);

    void setGrad(Tensor<double> grad);
    Tensor<double> getGrad();

    void set_grad_1();

    shared_ptr<Value> sum();

    shared_ptr<Value> mean();

    shared_ptr<Value> tanh();
    shared_ptr<Value> neg();

    shared_ptr<Value> operator+(const double &other);
    shared_ptr<Value> operator+(const shared_ptr<Value> &other);

    shared_ptr<Value> operator*(const double &other);
    shared_ptr<Value> operator*(const shared_ptr<Value> &other);

    shared_ptr<Value> operator-(const double &other);
    shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);

    shared_ptr<Value> operator/(const double &other);
    shared_ptr<Value> operator/(const shared_ptr<Value> &other);

    shared_ptr<Value> dot(const shared_ptr<Value> &other);

    // String representation
    friend std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v);
    void backward();
    
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const double& rhs);
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const double& rhs);
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);    

    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const double& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);  
    
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const double& rhs);
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);

};

#endif