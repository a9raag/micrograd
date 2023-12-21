#include <iostream>
#include <functional>
#include <set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <string>
#include <engine.h>
using namespace std;

// Constructor
Value::Value(double data, std::initializer_list<shared_ptr<Value>> children = {}, std::string op = "", std::string label = "")
    : data(data), grad(0.0), node_backward([]() {}), prev(children), _op(op), label(label) {}
void Value::set_data(double data)
{
    this->data = data;
}
void Value::set_grad(double grad)
{
    this->grad = grad;
}
double Value::get_data()
{
    return this->data;
}
double Value::get_grad()
{
    return this->grad;
}

Value::~Value() {
    // cout<<"deleting "<<this->label<<endl;
}
shared_ptr<Value> Value::pow(float n)
{
    auto out = make_shared<Value>(std::pow(this->data, n), std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "pow");
    out->node_backward = [this, &out, n]() mutable
    {
        this->grad += n * std::pow(this->data, n - 1) * out->grad;
    };
    return out;
}


shared_ptr<Value> Value::tanh()
{
    float x = this->data;
    float t = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    auto out = make_shared<Value>(std::tanh(this->data), std::initializer_list<std::shared_ptr<Value>>{shared_from_this()}, "tanh");
    out->node_backward = [this, &out, t]() mutable
    {
        this->grad += (1.0 - std::pow(t, 2)) * out->grad;
    };
    return out;
}
// Overload the + operator
shared_ptr<Value> Value::operator+(const shared_ptr<Value> &other)
{
    auto out = make_shared<Value>(this->data + other->data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
    out->node_backward = [this, &out, &other]() mutable
    {
        this->grad += 1.0 * out->grad;
        other->grad += 1.0 * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::operator*(const shared_ptr<Value> &other)
{
    auto out = make_shared<Value>(this->data * other->data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
    out->node_backward = [this, &out, &other]() mutable
    {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}

shared_ptr<Value> Value::neg()
{
    return shared_from_this();
}

shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return shared_from_this() + other->neg();
}


shared_ptr<Value> Value::operator/(const shared_ptr<Value> &other)
{
    return shared_from_this() * other->pow(-1.0);
}

// bool operator<(const shared_ptr<Value> &other) const
// {
//     return shared_from_this() < (&other);
// }
// String representation
std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v)
{
    os << "Value(data=" << v->data << ", grad=" << v->grad << ", op=" << v->_op << ")";
    return os;
}

void Value::backward()
{
    std::list<shared_ptr<Value>> topo;
    std::set<shared_ptr<Value>> visited;
    visited.end();
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
    this->grad = 1.0;

    build_topo(shared_from_this());
    for (auto v : topo)
    {
        v->node_backward();
    }
}



std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);  
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& lhs, const int n);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);


std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& lhs, const int n) {
    return lhs->pow(n);
}
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs->neg();
}


std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs->pow(-1.0);
}

void test_backprop()
{
    // Example usage
    shared_ptr<Value> x1 = std::make_shared<Value>(2.0);
    shared_ptr<Value> x2 = std::make_shared<Value>(0.0);
    x1->label = "x1";
    x2->label = "x2";
    shared_ptr<Value> w1 = std::make_shared<Value>(-3.0);
    shared_ptr<Value> w2 = std::make_shared<Value>(0.0);
    w1->label = "w1";
    w2->label = "w2";
    shared_ptr<Value> b = std::make_shared<Value>(6.8813735870195432);
    b->label = "b";
    shared_ptr<Value> x1w1 = x1 * w1;
    x1w1->label = "x1w1";
    shared_ptr<Value> x2w2 = x2 * w2;
    x2w2->label = "x2w2";
    shared_ptr<Value> x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label = "x1w1x2w2";
    shared_ptr<Value> n = x1w1x2w2 + b;
    n->label = "n";
    shared_ptr<Value> o = n->tanh();
    o->label = "o";

    // Value c = o.pow(2.0);
    // c.label = 'c';

    std::cout << "Starting backward pass" << std::endl;
    o->backward();
    // cout << "c: " << c << endl;
    // cout<<"c: "<<c.grad<<endl;
    cout << "o: " << o << endl;
    cout << "n: " << n << endl;
    cout << "x1w1x2w2: " << x1w1x2w2 << endl;
    cout << "x2w2: " << x2w2<< endl;
    cout << "x1w1: " << x1w1 << endl;
    cout << "b: " << b << endl;
    cout << "w2: " << w2 << endl;
    cout << "x2: " << x2 << endl;
    cout << "w1: " << w1 << endl;
    cout << "x1: " << x1 << endl;
}
int main(int argc, char const *argv[])
{
    test_backprop();
    return 0;
}
