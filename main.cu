#include <iostream>
#include<vector>
#include "engine.cu"
using namespace std;
int main(){
    double* arr = new double[5];
    vector<double> v1 = {1,2,3,4,5};
    shared_ptr<Value> a = make_shared<Value>(v1.data(), v1.size());
    vector<double> v2 = {2.0, 3.0, 4.0, 5.0, 6.0};
    shared_ptr<Value> b = make_shared<Value>(v2.data(), v2.size());
    cout <<"a:"<<a<<endl;
    cout <<"b:"<<b<<endl;
    shared_ptr<Value> c = a + b;
    cout<<"c:"<<c<<endl;
    shared_ptr<Value> e = c->pow(2);
    e->set_grad_1();
    
    cout<<"e:"<<e<<endl;
    
    e->node_backward(); 
    c->node_backward();
    cout <<"a:"<<a<<endl;
    cout <<"b:"<<b<<endl;
    cout<<"c:"<<c<<endl;
    shared_ptr<Value> d = a * b; 
    d->set_grad_1();
    d->node_backward();
    cout <<"d:"<< d << endl;
    // cout << "Hello, from micrograd!\n";

    test_backprop();
    return 0;
}
