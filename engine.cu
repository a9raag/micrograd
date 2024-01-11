#include <iostream>
#include <functional>
#include <set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <string>
#include "./include/engine.h"
#include <cuda_runtime.h>
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
// Constructor
Value::Value(double* dataVal, int size, std::initializer_list<shared_ptr<Value>> children = {}, string op = "", string label = "")
    : data(dataVal), size(size),  prev(children), _op(op), label(label) {
        this->node_backward = []() {};
        this->threadsPerBlock = 256;
        this->blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;    

        // Initialize data on GPU
        cudaMalloc(&this->data, sizeof(double) * size);
        
        cudaMemcpy(this->data, dataVal, sizeof(double) * size, cudaMemcpyHostToDevice);

            // Initialize grad to 0 on GPU
        cudaMalloc(&grad, sizeof(double) * size );
        cudaMemset(grad, 0.0, sizeof(double) * size );
        cudaDeviceSynchronize();        
};

Value::~Value() {
    cudaFree(data);
    cudaFree(grad); 
}
//move constructor
Value::Value(Value &&other) noexcept
    : data(other.data), grad(other.grad), node_backward(move(other.node_backward)), prev(move(other.prev)), _op(move(other._op)), label(move(other.label)) {}   
void Value::set_data(double* data)
{
    this->data = data;
}
void Value::set_grad(double* grad)
{
    this->grad = grad;
}
double* Value::get_data()
{
    return this->data;
}
double* Value::get_grad()
{
    return this->grad;
}
__global__ void set_ones(double* grad, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] = 1.0;
    }
}
void Value::set_grad_1(){
    set_ones<<<blocksPerGrid, threadsPerBlock>>>(this->grad, this->size);
    cudaDeviceSynchronize();
}
__global__ void cudapow(double* a, double *b, double n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        b[i] = pow(a[i], n);
    }
}

__global__ void cudaPowGrad(double* data, double *outGrad, double* grad, double n, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += n * pow(data[i], n - 1) * outGrad[i];
    }
}
shared_ptr<Value> Value::pow(double n)
{

    double *doutData;
    cudaMalloc(&doutData, sizeof(double) * this->size);
    cudaMemset(doutData, 0, sizeof(double) * this->size);
    cudapow<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), doutData, n, this->size);

    vector<double> houtData(this->size);
    cudaMemcpy(houtData.data(), doutData, size * sizeof(double), cudaMemcpyDeviceToHost);
    auto out = make_shared<Value>(houtData.data(), size, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()});
    out->node_backward = [this, out, n]() mutable
    {
        cudaPowGrad<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), out->get_grad(), this->get_grad(), n, this->size);
        cudaDeviceSynchronize();
    };
    return out;
}
__global__ void cudaTanh(double* data, double * out, int size){ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size){
        double x = data[i]; 
        out[i] = (exp(2.0 * x) - 1.0) / (exp(2.0 * x) + 1.0);
    }
}

__global__ void gradTanh(double* grad, double* outGrad, double* data, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double t = (exp(2.0 * data[i]) - 1.0) / (exp(2.0 * data[i]) + 1.0);
        grad[i] += (1.0 - pow(t, 2)) * outGrad[i];
    }
}

shared_ptr<Value> Value::tanh()
{
    double *doutData;
    cudaMalloc(&doutData, sizeof(double) * this->size);
    cudaTanh<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), doutData, this->size);

    vector<double> houtData(this->size);
    cudaMemcpy(houtData.data(), doutData, size * sizeof(double), cudaMemcpyDeviceToHost);
    auto out = make_shared<Value>(houtData.data(), size, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()});
    out->node_backward = [this, out]() mutable
    {
        gradTanh<<<blocksPerGrid, threadsPerBlock>>>(this->grad, out->get_grad(), this->get_data(), this->size);
        cudaDeviceSynchronize();
    };
    return out;
}
// // Overload the + operator
__global__ void add(double* a, double* b, double* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

__global__ void addGrad(double* grad, double* outGrad, double* otherGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += 1.0 * outGrad[i];
        otherGrad[i] += 1.0 * outGrad[i];
    }
}
shared_ptr<Value> Value::operator+(const shared_ptr<Value> &other)
{
    double *doutData;
    cudaMalloc(&doutData, sizeof(double) * this->size);
    add<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), other->get_data(), doutData, this->size);
    cudaDeviceSynchronize();

    vector<double> houtData(this->size);
    cudaMemcpy(houtData.data(), doutData, size * sizeof(double), cudaMemcpyDeviceToHost);

    
    auto out = make_shared<Value>(houtData.data(), this->size, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()});
    // auto out = make_shared<Value>(this->data + other->data, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
    out->node_backward = [this, out, other]() mutable
    {
        // this->grad += 1.0 * out->grad;
        // other->grad += 1.0 * out->grad;
        addGrad<<<blocksPerGrid, threadsPerBlock>>>(grad, out->get_grad(), other->get_grad(), this->size);
        cudaDeviceSynchronize();
    };
    return out;
}

__global__ void dot(double* data, double* other, double* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = data[i] * other[i];
    }
}

__global__ void dot(double* data, double other, double* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = data[i] * other;
    }
}

__global__ void dotGrad(double* grad, double * data, double* outGrad, double* otherData, double* otherGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += otherData[i] * outGrad[i];
        otherGrad[i] += data[i] * outGrad[i];
    }
}
shared_ptr<Value> Value::operator*(const shared_ptr<Value> &other)
{
    double *doutData;
    cudaMalloc(&doutData, sizeof(double) * this->size);
    dot<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), other->get_data(), doutData, this->size);
    cudaDeviceSynchronize();
    vector<double> houtData(this->size);
    cudaMemcpy(houtData.data(), doutData, size * sizeof(double), cudaMemcpyDeviceToHost);

    auto out = make_shared<Value>(houtData.data(), this->size, std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other});
    // auto out = make_shared<Value>(houtData.data(), std::initializer_list<std::shared_ptr<Value>>{shared_from_this(), other});
    out->node_backward = [this, out, other]() mutable
    {
        // this->grad += other->data * out->grad;
        // other->grad += this->data * out->grad;
        dotGrad<<<blocksPerGrid, threadsPerBlock>>>(this->grad, this->get_data(), out->get_grad(), other->get_data(), other->get_grad(), this->size);
        cudaDeviceSynchronize();
    };
    return out;
}

shared_ptr<Value> Value::neg()
{
    double *doutData; 
    cudaMalloc(&doutData, sizeof(double) * this->size);
    dot<<<blocksPerGrid, threadsPerBlock>>>(this->get_data(), -1.0, doutData, this->size);
    cudaDeviceSynchronize();
    vector<double> houtData(this->size);
    cudaMemcpy(houtData.data(), doutData, size * sizeof(double), cudaMemcpyDeviceToHost);
    auto out = make_shared<Value>(houtData.data(), this->size, std::initializer_list<std::shared_ptr<Value>>{shared_from_this()});
    return out; 
}

shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return shared_from_this() + other->neg();
}


shared_ptr<Value> Value::operator/(const shared_ptr<Value> &other)
{
    return shared_from_this() * other->pow(-1.0);
}

// // bool operator<(const shared_ptr<Value> &other) const
// // {
// //     return shared_from_this() < (&other);
// // }

// // String representation
std::ostream &operator<<(std::ostream &os, const shared_ptr<Value> &v)
{

    vector<double> houtData(v->size);
    cudaMemcpy(houtData.data(), v->data, sizeof(double) * v->size, cudaMemcpyDeviceToHost);
    string arr = to_string(houtData.data(), v->size);
    vector<double> houtData2(v->size);
    cudaMemcpy(houtData2.data(), v->grad, sizeof(double) * v->size, cudaMemcpyDeviceToHost);
    string grad_arr = to_string(houtData2.data(), v->size);
    os << "Value(data=" << arr << ", grad=" << grad_arr << ", op=" << v->_op << ")";
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
    this->set_grad_1();

    build_topo(shared_from_this());
    for (auto v : topo)
    {
        v->node_backward();
    }
}


std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    if (lhs == NULL || rhs == NULL){
        // cout<<lhs<<" "<<rhs<<endl;
        cout<<"lhs or rhs are null"<<endl;
        // throw "lhs or rhs is null";
    }
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
    vector<double> x1v = {2.0};
    shared_ptr<Value> x1 = std::make_shared<Value>(x1v.data(), x1v.size());
    vector<double> x2v = {1.0};
    shared_ptr<Value> x2 = std::make_shared<Value>(x2v.data(), x2v.size());
    x1->label = "x1";
    x2->label = "x2";
    vector<double> w1v = {-3.0};
    vector<double> w2v = {0.0};
    shared_ptr<Value> w1 = std::make_shared<Value>(w1v.data(), w1v.size());
    shared_ptr<Value> w2 = std::make_shared<Value>(w2v.data(), w2v.size());
    w1->label = "w1";
    w2->label = "w2";
    vector<double> bv = {6.8813735870195432};
    shared_ptr<Value> b = std::make_shared<Value>(bv.data(), bv.size());
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
// int main(int argc, char const *argv[])
// {
//     cout<<"Cuda"<<endl;
//     return 0;
// }
