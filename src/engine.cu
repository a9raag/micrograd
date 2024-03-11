#include "include/engine.h"
#include <iostream>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <string>
#include "engine.h"


using namespace std;
string to_string(float* data, int size){
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
Value::Value() : data(Tensor<float>()), node_backward([](){}), prev({}), op(""), label("") {
    this->node_backward = []() {};
    this->grad = Tensor<float>();
}

template<typename T>
Value::Value(initializer_list<T> input)
{
    vector<T> input_vector = input;
    Tensor<float>  data = Tensor<float>({input_vector.size()});
    cout<<data(0)<<endl;
    for(int i = 0; i < input.size(); i++){
        data(i) = input_vector[i];
    }   
    this->node_backward = []() {};
    this->grad = Tensor<float>(data.shape);

}

Value::Value(Tensor<float> data, std::initializer_list<shared_ptr<Value>> children = {}, string op = "", string label = "")
    : data(data), node_backward([](){}), prev(children), op(op), label(label) {
        this->node_backward = []() {}; 
        this->grad = Tensor<float>(data.shape);    
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

void Value::zero_grad(){
    this->grad.fill(0.0);
}

shared_ptr<Value> Value::subTensor(vector<vector<size_t>> dimRanges)
{
    Tensor<float> subData = this->data.subTensor(dimRanges);
    auto out = make_shared<Value>(subData, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "slice", label);
    out->node_backward = [this, out]() mutable
    {
        this->grad = this->grad + out->grad;
    };
    return out;
}

shared_ptr<Value> Value::sum()
{
    Tensor<float> sum_data = this->data.sum();
    auto out = make_shared<Value>(sum_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "sum", label);
    
    out->node_backward = [this, out]() mutable
    {
        this->grad = this->grad + out->grad;
    };
    return out;
}

shared_ptr<Value> Value::sum(int axis){
    Tensor<float> sum_data = this->data.sum(axis);
    auto out = make_shared<Value>(sum_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "sum", label);
    out->node_backward = [this, out]() mutable
    {
        this->grad = this->grad + out->grad;
    };
    return out;
}

shared_ptr<Value> Value::mean()
{
    return this->sum() * (1.0/ (float) this->data.size);
}

Tensor<float> Value::getData(){
    return this->data;
}

void Value::setData(Tensor<float> data)
{
    this->data = data;
}
Tensor<float> Value::getGrad()
{
    return this->grad;
}

shared_ptr<Value> Value::neg(){
    Tensor<float> neg_data = Tensor<float>(this->data.shape);
    neg_data.fill(-1.0);
    auto out = make_shared<Value>(neg_data, std::initializer_list<std::shared_ptr<Value>>{}, "neg", label);
    return shared_from_this() * out;
}

shared_ptr<Value> Value::exp()
{
    Tensor<float> exp_data = this->data.exp();
    auto out = make_shared<Value>(exp_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "exp", label);
    out->node_backward = [this, out, exp_data]() mutable
    {
        this->grad = this->grad + out->grad * exp_data;
    };
    return out;
}

shared_ptr<Value> Value::log()
{
    Tensor<float> log_data = this->data.log();
    auto out = make_shared<Value>(log_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "log", label);
    out->node_backward = [this, out]() mutable
    {
        this->grad = this->grad + out->grad * this->getData().pow(-1);
    };
    return out;
}

shared_ptr<Value> Value::sigmoid()
{
    Tensor<float> sigmoid_data = this->data.sigmoid();
    auto out = make_shared<Value>(sigmoid_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "sigmoid", label);
    out->node_backward = [this, out, sigmoid_data]() mutable
    {
        auto ones = Tensor<float>(this->data.shape);
        ones.fill(1.0);
        auto grad = sigmoid_data * (ones - sigmoid_data) * out->grad;
        this->grad = this->grad + grad;
    };
    return out;
}

shared_ptr<Value> Value::relu()
{
    Tensor<float> relu_data = this->data.relu();
    auto out = make_shared<Value>(relu_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "relu", label);
    out->node_backward = [this, out]() mutable
    {
        auto grad = out->grad * (this->data > 0);
        this->grad = this->grad + grad;
    };
    return out;
}

shared_ptr<Value> Value::pow(const float n)
{
    Tensor<float> pow_data = Tensor<float>(this->data.shape);
    pow_data = this->data.pow(n);
    auto out = make_shared<Value>(pow_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "pow", label);
    out->node_backward = [this, out, n]() mutable
    {
        this->grad = this->grad +  out->grad * n * this->data.pow(n - 1);
    };
    return out;
}

shared_ptr<Value> Value::tanh(){
    Tensor<float> tanh_data = this->data.tanh();
    auto out = make_shared<Value>(tanh_data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "tanh", label);
    out->node_backward = [this, out, tanh_data]() mutable
    {
        Tensor<float> ones = Tensor<float>(this->data.shape);
        ones.fill(1.0);
        auto t2 = tanh_data.pow(2);
        auto grad = (ones - t2)* out->grad;
        this->grad = this->grad +  grad;
    };
    return out;
}

shared_ptr<Value> Value::operator+(const float &other)
{
    Tensor<float> other_tensor = Tensor<float>(this->data.shape).fill(other);
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
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const float &rhs)
{
    return (*lhs) + rhs;
}
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs)
{
    return (*lhs) + rhs;
}

shared_ptr<Value> Value::operator-(const float &other)
{
    Tensor<float> other_data = Tensor<float>(this->data.shape);
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

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const float& rhs) {
    return (*lhs) - rhs;
}

shared_ptr<Value> Value::operator*(const float &other)
{
    Tensor<float> other_data = Tensor<float>(this->data.shape);
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


std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const float& rhs) {
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

shared_ptr<Value> Value::operator/(const float &other)
{
    Tensor<float> other_data = Tensor<float>(this->data.shape);
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

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const float& rhs) {
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
