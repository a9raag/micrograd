#pragma once
#ifndef ENGINE_H
#define ENGINE_H

#include <iostream>
#include <functional>
#include <set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <string>
using namespace std;

class Value: public std::enable_shared_from_this<Value>
{
private:
    double *data;
    double *grad;
    int size;
    
    set<shared_ptr<Value>> prev;
    std::string _op;
    int threadsPerBlock;
    int blocksPerGrid;
public:
    function<void()> node_backward;
    std::string label;
    // Constructor
    Value(double *data, int size, std::initializer_list<shared_ptr<Value>> children, string op, string label);
    //move constructor
    Value(Value &&other) noexcept;
    ~Value();
    shared_ptr<Value> pow(double n);
    double* get_data();
    void set_data(double *data);
    void set_grad(double *grad);
    double* get_grad();
    void set_grad_1();
    shared_ptr<Value> tanh();
    shared_ptr<Value> operator+(const shared_ptr<Value> &other);
    shared_ptr<Value> operator*(const shared_ptr<Value> &other);
    shared_ptr<Value> neg();
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