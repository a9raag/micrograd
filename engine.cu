#include "include/engine.h"
#include <iostream>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <string>

using namespace std;
string to_string(double* data, int size){
    string arr = "["; 
    for(int i = 0; i < size; i++){
        arr += to_string(data[i]);
        if(i != size - 1){
            arr += ", ";
        }
    }
    arr += "]";
    return arr;
}
// default constructor
Value::Value() : data(Tensor<double>()), node_backward([](){}), prev({}), op(""), label("") {
    this->node_backward = []() {};
    this->grad = Tensor<double>();
}


Value::Value(Tensor<double> data, std::initializer_list<shared_ptr<Value>> children = {}, string op = "", string label = "")
    : data(data), node_backward([](){}), prev(children), op(op), label(label) {
        this->node_backward = []() {}; 
        this->grad = Tensor<double>(data.shape);    
};

Value::~Value() {
    // delete grad;
}
//copy constructor
Value::Value(const Value &other)
    : data(other.data), grad(other.grad), node_backward(other.node_backward), prev(other.prev), op(other.op), label(other.label) {}

//move constructor
Value::Value(Value &&other) noexcept
    : data(move(other.data)), grad(move(other.grad)), node_backward(move(other.node_backward)), prev(move(other.prev)), op(move(other.op)), label(move(other.label)) {}   


void Value::set_grad_1(){
    this->grad.fill(1.0);
}


Tensor<double> Value::getData(){
    return this->data;
}

Tensor<double> Value::getGrad(){
    return this->grad;
}

shared_ptr<Value> Value::neg(){
    Tensor<double> neg_data = Tensor<double>(this->data.shape);
    neg_data.fill(-1.0);
    auto out = make_shared<Value>(neg_data, std::initializer_list<std::shared_ptr<Value>>{}, "neg", label);
    return shared_from_this() * out;
}

shared_ptr<Value> Value::pow(const double n){
    Tensor<double> pow_data = Tensor<double>(this->data.shape);
    pow_data = this->data.pow(n);
    auto out = make_shared<Value>(pow_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "pow", label);
    out->node_backward = [this, out, n]() mutable
    {
        this->grad = this->grad +  out->grad * n * this->data.pow(n - 1);
    };
    return out;
}

shared_ptr<Value> Value::tanh(){
    Tensor<double> tanh_data = this->data.tanh();
    auto out = make_shared<Value>(tanh_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "tanh", label);
    out->node_backward = [this, out, tanh_data]() mutable
    {
        Tensor<double> ones = Tensor<double>(this->data.shape);
        ones.fill(1.0);
        auto tanh_2 = tanh_data.pow(2);
        auto tanh_grad = ones - tanh_2;
        auto grad = tanh_grad * out->grad;
        this->grad = this->grad +  grad;
    };
    return out;
}

shared_ptr<Value> Value::operator+(const double &other)
{
    Tensor<double> other_tensor = Tensor<double>(this->data.shape).fill(other);
    auto other_val = make_shared<Value>(other_tensor, std::initializer_list<std::shared_ptr<Value>>{}, "const", label);
    return shared_from_this() + other_val;
}

shared_ptr<Value> Value::operator+(const shared_ptr<Value> &other)
{
    
    auto out = make_shared<Value>(data + other->data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "+", label);
    out->node_backward = [this, out, other]() mutable
    {  
        this->grad = this->grad +  out->grad * 1.0;
        other->grad = other->grad + out->grad * 1.0;
    };
    return out;
    
}
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const double &rhs)
{
    return (*lhs) + rhs;
}
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    return (*lhs) + rhs;
}

shared_ptr<Value> Value::operator-(const double &other)
{
    Tensor<double> other_data = Tensor<double>(this->data.shape);
    other_data.fill(other);
    auto other_val = make_shared<Value>(other_data, std::initializer_list<std::shared_ptr<Value>>{}, "const", label);
    return shared_from_this() - other_val;
}

shared_ptr<Value> Value::operator-(const shared_ptr<Value> &other)
{
    return shared_from_this() + other->neg();
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) - rhs;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const double& rhs) {
    return (*lhs) - rhs;
}

shared_ptr<Value> Value::operator*(const double &other)
{
    Tensor<double> other_data = Tensor<double>(this->data.shape);
    other_data.fill(other);
    auto other_val = make_shared<Value>(other_data, std::initializer_list<std::shared_ptr<Value>>{}, "const", label);
    return shared_from_this() * other_val;
}

shared_ptr<Value> Value::operator*(const shared_ptr<Value> &other)
{
    auto out = make_shared<Value>(data * other->data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "*", label);
    out->node_backward = [this, out, other]() mutable
    {
        this->grad = this->grad + out->grad * other->data;
        other->grad = other->grad + out->grad * this->data;
    };
    return out;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}


std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const double& rhs) {
    return (*lhs) * rhs;
}




shared_ptr<Value> Value::dot(const shared_ptr<Value> &other)
{
    auto out = make_shared<Value>(data.dot(other->data), std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "dot", label);
    out->node_backward = [this, out, other]() mutable
    {
        auto otherDataT = other->data.transpose();
        this->grad = this->grad +  out->grad.dot(otherDataT);
        other->grad = other->grad + this->data.transpose().dot(out->grad);
    };
    return out;
}

shared_ptr<Value> Value::operator/(const double &other)
{
    Tensor<double> other_data = Tensor<double>(this->data.shape);
    other_data.fill(other);
    auto other_val = make_shared<Value>(other_data, std::initializer_list<std::shared_ptr<Value>>{}, "const", label);
    return shared_from_this() / other_val;
}

shared_ptr<Value> Value::operator/(const shared_ptr<Value> &other)
{
    return shared_from_this() * other->pow(-1);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) / rhs;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const double& rhs) {
    return (*lhs) / rhs;
}

// String representation
std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v)
{
    //handle null ptr 
    if(v == NULL){
        throw "v is null";
    }
    os << "Value(data=" << v->getData() << ", grad=" << v->getGrad() << ", op=" << v->op << ")";
    return os;
}

void Value::backward()
{
    std::list<shared_ptr<Value>> topo;
    std::set<shared_ptr<Value>> visited;
    std::function<void(const shared_ptr<Value>&)> build_topo = [&](const shared_ptr<Value> &v)
    {
        if (visited.count(v) == 0)
        {
            visited.insert(v);
            for (auto child : v->prev)
            {
                build_topo(child);
            }
            topo.push_front(v);
        }
    };
    this->grad.fill(1.0);

    build_topo(shared_from_this());
    for (auto v : topo)
    {
        v->node_backward();
    }
}
