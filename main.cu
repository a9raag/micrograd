#include <iostream>
#include<vector>
#include "tensor.cu"
#include "compute1d.cu"
#include "compute2d.cu"
#include "engine.cu"
#include <cuda_runtime.h>
#include "nn.cpp"
using namespace std;

void test_compute(){
    
    vector<int> _a = {1, 2, 3};
    vector<int> _b = {2, 3, 4};

    Compute1D<int> a = Compute1D<int>(_a, _a.size());
    Compute1D<int> b = Compute1D<int>(_b, _b.size());

    int * c;
    size_t* shape = new size_t[1];
    shape[0] = 3;
    c = a.add(b);

    cout<<"C:"<<endl;
    for(int i = 0; i < _a.size(); i++){
        cout<<c[i]<<endl;
    }
}

void test_tensor_1d(){
    cout<<"=========================="<<endl;
    cout<<"START: Testing Tensor 1d"<<endl;
    cout<<"=========================="<<endl;

    Tensor<double> a({3}), b({3});
    a(0) = 1;
    a(1) = 2; 
    a(2) = 3;

    b(0) = 4;
    b(1) = 5;
    b(2) = 6;

    cout<<"a:"<<endl;
    cout<<a<<endl;

    cout<<"b:"<<endl;
    cout<<b<<endl;

    auto c = a+b;
    cout<<"c:"<<endl;
    cout<<c<<endl;

    c = a * b;
    cout<<"c:"<<endl;
    cout<<c<<endl;

    auto d = c - a;
    cout<<"c:"<<endl;
    cout<<d<<endl;

    d = b / a;
    cout<<"b/a"<<endl;
    cout<<d<<endl;

    cout<<"a**2"<<endl;
    cout<<a.pow(2.0)<<endl;

    d = a.dot(b);
    cout<<"a@b"<<endl;
    cout<<d<<endl;

    Tensor<double> e = Tensor<double>({1});
    e(0) = 2.0;
    d = d + e;
    cout<<"d+e"<<endl;
    cout<<d<<endl;

    cout<<"END: Testing Tensor 1d"<<endl;
}
void test_tensor(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Tensor: a is 3x3, b is 3x3"<<endl;
    cout<<"=========================="<<endl;
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
    cout<<"END: Testing Tensor"<<endl;
    
}
void test_tensor_2d(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Tensor 2d: a is 2x3, b is 3x2"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a({2, 3});
    Tensor<double> b({3, 2});
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(0, 2) = 3;
    a(1, 0) = 4;
    a(1, 1) = 5;
    a(1, 2) = 6;


    b(0, 0) = 20;
    b(0, 1) = 30;
    b(1, 0) = 40;
    b(1, 1) = 50;
    b(2, 0) = 60;
    b(2, 1) = 70;
    
    cout<<"a:"<<endl;
    cout<<a<<endl;
    cout<<"b:"<<endl;
    cout<<b<<endl;

    // auto c = a+b;
    // cout<<"c:a+b"<<endl;
    // cout<<c<<endl;

    // c = a * b;
    // cout<<"c:a*b"<<endl;
    // cout<<c<<endl;

    auto c = a.dot(b);
    cout<<"c:a@b"<<endl;    
    cout<<c<<endl;

    cout<<"END: Test Tensor 2d"<<endl;

}
void test_value2d(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Value 2d: a is 3x3, b is 3x3"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a({3, 3}), b({3, 3});

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
    cout<<"a"<<endl;
    cout<<a<<endl;
    cout<<"--------------------------"<<endl;

    cout<<"b"<<endl;
    cout<<b<<endl;
    cout<<"--------------------------"<<endl;

    auto c = a + b;
    cout<<"c: a + b"<<endl;
    cout<<c<<endl;
    cout<<"--------------------------"<<endl;

    auto d = a * b;
    cout<<"d: a * b"<<endl;
    cout<<d<<endl;
    cout<<"--------------------------"<<endl;

    shared_ptr<Value> val_a = std::make_shared<Value>(a);
    shared_ptr<Value> val_b = std::make_shared<Value>(b);

    auto val_c = val_a + val_b;
    cout<<"val_c: val_a + val_b"<<endl;
    cout<<val_c->getData()<<endl;
    cout<<"--------------------------"<<endl;

    auto val_d = val_a * val_b;
    cout<<"val_d: val_a * val_b"<<endl;
    cout<<val_d->getData()<<endl;
    cout<<"--------------------------"<<endl;

    auto val_e = val_a->dot(val_b);
    cout<<"val_e: val_a.dot(val_b)"<<endl;
    cout<<val_e->getData()<<endl;
    cout<<"--------------------------"<<endl;


    val_c = val_c->pow(0.5);
    cout<<"val_c: val_c ** 0.5"<<endl;
    cout<<val_c->getData()<<endl;
    cout<<"--------------------------"<<endl;

    val_c = val_c->tanh();
    cout<<"val_c: val_c.tanh()"<<endl;
    cout<<val_c->getData()<<endl;
    cout<<"--------------------------"<<endl;

    val_c->set_grad_1();
    val_c->node_backward();
    cout<<"val_a:"<<endl;
    cout<<val_a->getGrad()<<endl;
    cout<<"--------------------------"<<endl;
    
    cout<<"END: Test Value 2d: a is 3x3, b is 3x3"<<endl;
    

}

void test_backprop(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Backprop"<<endl;
    cout<<"=========================="<<endl;
    cout << "Starting forward pass" << endl;
    Tensor<double> xt1 = Tensor<double>({1});
    xt1(0) = 2.0;
    shared_ptr<Value> x1 = std::make_shared<Value>(xt1);
    Tensor<double> xt2 = Tensor<double>({1});
    xt2(0) = 0.0;
    shared_ptr<Value> x2 = std::make_shared<Value>(xt2);
    x1->label = "x1";
    x2->label = "x2";
    Tensor<double> wt1 = Tensor<double>({1});
    wt1(0) = -3.0;
    shared_ptr<Value> w1 = std::make_shared<Value>(wt1);
    Tensor<double> wt2 = Tensor<double>({1});
    wt2(0) = 0.0;
    shared_ptr<Value> w2 = std::make_shared<Value>(wt2);
    w1->label = "w1";
    w2->label = "w2";

    Tensor<double> bt = Tensor<double>({1});
    bt(0) = 6.8813735870195432;
    shared_ptr<Value> b = std::make_shared<Value>(bt);
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
    cout<<"END: Test Backprop"<<endl;
}
void test_random(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Random"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a = Tensor<double>({3, 3}).randomize();
    
    cout<<"a"<<endl;
    cout<<a.randomize()<<endl;

    Tensor<double> b ({10});
    b.randomize();
    cout<<"b"<<endl;
    cout<<b<<endl;

    Tensor<double> c ({1, 10});
    c.randomize();
    cout<<"c"<<endl;
    cout<<c<<endl;
    cout<<"END: Test Random"<<endl;
}

void test_value_broadcast(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Value Broadcast"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a = Tensor<double>({3, 3});
    Tensor<double> b = Tensor<double>({3, 3});
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
    



    cout<<"a"<<endl;
    cout<<a<<endl;
    cout<<"b"<<endl;
    cout<<b<<endl;
    auto c = a + b;
    cout<<"c"<<endl;
    cout<<c<<endl;

    auto val_a = std::make_shared<Value>(a);
    auto val_b = std::make_shared<Value>(b);
    auto val_c = val_a + 1.0;
    cout<<"val_c: val_a + 1.0"<<endl;
    cout<<val_c->getData()<<endl;

    Tensor<double> row = Tensor<double>({1, 3});
    row(0,0) = 1;
    row(0,1) = 2;
    row(0,2) = 3;
    auto val_row = std::make_shared<Value>(row);
    cout<<"val_row: "<<val_row->getData()<<endl;
    auto val_d = val_a + val_row * 10;
    cout<<"val_d: val_a + val_row"<<endl;
    cout<<val_d->getData()<<endl;

    Tensor<double> column = Tensor<double>({3, 1});
    column(0,0) = 100;
    column(1,0) = 200;
    column(2,0) = 300;

    auto val_column = std::make_shared<Value>(column);
    cout<<"val_column: \n"<<val_column->getData()<<endl;
    auto val_e = val_a + val_column;
    cout<<"val_e: val_a + val_column"<<endl;
    cout<<val_e->getData()<<endl;

    auto val_f = val_a->dot(val_column);
    cout<<"val_f: val_a.dot(val_column)"<<endl;
    cout<<val_f->getData()<<endl;

    cout<<"END: Test Value Broadcast"<<endl;

}
void test_gradient(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Gradient"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a = Tensor<double>({3, 3});
    Tensor<double> b = Tensor<double>({3, 3});
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
    cout<<"a"<<endl;
    cout<<a<<endl;
    cout<<"b"<<endl;
    cout<<b<<endl;

    auto val_a = std::make_shared<Value>(a);
    auto val_b = std::make_shared<Value>(b);
    auto val_c = val_a + val_b;
    cout<<"val_c: val_a + val_b"<<endl;
    cout<<val_c->getData()<<endl;

    val_c->set_grad_1();
    val_c->backward();
    cout<<"val_a grad"<<endl;
    cout<<val_a->getGrad()<<endl;
    cout<<"val_b grad"<<endl;
    cout<<val_b->getGrad()<<endl;

    
    val_a = std::make_shared<Value>(a);
    val_b = std::make_shared<Value>(b);
    auto val_d = val_a->dot(val_b);
    cout<<"val_d: val_a @ val_b"<<endl;
    cout<<val_d->getData()<<endl;

    val_d->set_grad_1();
    val_d->backward();
    cout<<"val_a grad"<<endl;
    cout<<val_a->getGrad()<<endl;
    cout<<"val_b grad"<<endl;
    cout<<val_b->getGrad()<<endl;


    val_a = std::make_shared<Value>(a);
    val_b = std::make_shared<Value>(b);
    val_d = val_a * val_b;
    cout<<"val_d: val_a * val_b"<<endl;
    cout<<val_d->getData()<<endl;

    val_d->set_grad_1();
    val_d->backward();
    cout<<"val_a grad"<<endl;
    cout<<val_a->getGrad()<<endl;
    cout<<"val_b grad"<<endl;
    cout<<val_b->getGrad()<<endl;

    cout<<"END: Test Gradient"<<endl;

}
void test_layer(){
    cout<<"=========================="<<endl;
    cout<<"START: Test Layer"<<endl;
    cout<<"=========================="<<endl;
    Tensor<double> a = Tensor<double>({1, 4});
    a(0, 0) = 2.0;
    a(0, 1) = 3.0;
    a(0, 2) = 4.0;
    a(0, 3) = 1.0;

    Layer layer = Layer({4, 2});
    shared_ptr<Value> x = std::make_shared<Value>(a);
    auto out = layer(x);
    out->set_grad_1();
    out->backward();
    cout<<"out: "<<out<<endl;
    cout<<"out grad: "<<out->getGrad()<<endl;
    cout<<"END: Test Layer"<<endl;
}

int main(int argc, char const *argv[]){
    test_tensor_1d();
    test_tensor_2d();
    test_value2d();
    test_backprop();
    test_gradient();
    test_random();
    test_value_broadcast();
    test_layer();
    return 0;

}