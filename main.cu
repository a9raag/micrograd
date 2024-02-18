#include <iostream>
#include<vector>
// #include "tensor.cu"
#include "engine.cu"
#include <cuda_runtime.h>
using namespace std;
void test_compute(){
    
    vector<int> _a = {1, 2, 3};
    vector<int> _b = {2, 3, 4};

    Compute1D<int> a = Compute1D<int>(_a, _a.size());
    Compute1D<int> b = Compute1D<int>(_b, _b.size());

    int * c;
    c = a.add(b.getData());

    cout<<"C:"<<endl;
    for(int i = 0; i < _a.size(); i++){
        cout<<c[i]<<endl;
    }
}
void test_tensor(){
    Tensor<double> a({3, 3});
    Tensor<double> b({3, 3});
    a(0,0) = 1;
    a(0,1) = 2;
    a(0,2) = 3;

    a(1,0) = 4;
    a(1,1) = 5;
    a(1,2) = 6;

    a(2,0) = 7;
    a(2,1) = 8;
    a(2,2) = 9;

    b(0,0) = 20;
    b(0,1) = 30;
    b(0,2) = 40;

    b(1,0) = 50;
    b(1,1) = 60;
    b(1,2) = 70;

    b(2,0) = 80;
    b(2,1) = 90;
    b(2,2) = 100;
    cout<<"a:"<<endl;
    cout<<a<<endl;
    cout<<"b:"<<endl;
    cout<<b<<endl;

    auto c = a+b; 
    cout<<"c:"<<endl;
    cout << c << endl;

    c = a * b;
    cout<<"c:"<<endl;
    cout << c <<endl;

    auto d = c - a;
    cout<<"c:"<<endl;
    cout << d << endl;
    
    d =  b / a; 
    cout<<"b/a"<<endl;
    cout << d << endl;

    cout<<"a**2"<<endl;
    cout<<a.pow(2.0)<<endl;

    cout<<"a.tanh()"<<endl;
    cout<<a.tanh()<<endl;
    
}

void test_value(){
    cout<<"Testing Value"<<endl;
    Tensor<double> a({3, 3});
    Tensor<double> b({3, 3});

    a(0,0) = 1;
    a(0,1) = 2;
    a(0,2) = 3;

    a(1,0) = 4;
    a(1,1) = 5;
    a(1,2) = 6;

    a(2,0) = 7;
    a(2,1) = 8;
    a(2,2) = 9;

    b(0,0) = 10;
    b(0,1) = 20;
    b(0,2) = 30;

    b(1,0) = 40;
    b(1,1) = 50;
    b(1,2) = 60;

    b(2,0) = 70;
    b(2,1) = 80;
    b(2,2) = 90;
    cout<<"a:"<<endl;
    auto c = a + b;
    cout<<"c:"<<endl;
    cout<<c<<endl;
    auto d = a * b;
    cout<<"d:"<<endl;
    cout<<d<<endl;

    auto val_a = make_shared<Value>(a);
    auto val_b = make_shared<Value>(b);
    auto val_c = val_a + val_b;

    cout<<"val_c:"<<endl;
    cout<<val_c->getData()<<endl;
    val_c = val_c->pow(0.5);
    cout<<"val_c:"<<endl;
    cout<<val_c->getData()<<endl;

    val_c = val_c->tanh();
    cout<<"tanh"<<endl;
    cout<<val_c->getData()<<endl;

    val_c->set_grad_1();
    val_c->node_backward();
    cout<<"val_a:"<<endl;
    cout<<val_a->getGrad()<<endl;

}
int main(int argc, char const *argv[]){
    // test_value();
    test_backprop();

    return 0;

}
