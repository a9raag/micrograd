#include <iostream>
#include <functional>
#include <set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include <engine.h>
using namespace std;
class Neuron
{
private: 
    std::vector<shared_ptr<Value>> weights;
    shared_ptr<Value> bias;
public:
    Neuron() {} // Remove the semicolon here
    int n_inputs;
    
    
    Neuron(int n_inputs)
    {

        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        n_inputs = n_inputs;
        weights.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            this->weights.push_back(make_shared<Value>(dis(gen)));
        }
         bias = make_shared<Value>(dis(gen));
    }

    shared_ptr<Value> operator()(std::vector<shared_ptr<Value>> inputs)
    {
        auto out = make_shared<Value>(0.0);
        for (int i = 0; i < this->n_inputs; i++)
        {
            auto val = (this->weights[i]) * inputs[i];
            out = out + val;
        }
       
        out = out + bias;
        return out;
    }

    std::vector<shared_ptr<Value>> get_params()
    {
        std::vector<shared_ptr<Value>> out;
        for (int i = 0; i < this->n_inputs; i++)
        {
            out.push_back(this->weights[i]);
        }
        out.push_back(this->bias);
        return out;
    }
};
// // class Layer
// // {
// // public:
// //     int n_inputs;
// //     int n_outs;
// //     std::vector<Neuron> neurons;
// //     Layer(int n_inputs, int n_outs)
// //     {
// //         this->n_inputs = n_inputs;
// //         this->n_outs = n_outs;
// //         for (int i = 0; i < n_outs; i++)
// //         {
// //             this->neurons.push_back(Neuron(n_inputs));
// //         }
// //     };

// //     std::vector<Value> operator()(std::vector<Value> &inputs)
// //     {
// //         std::vector<Value> out;
// //         for (auto neuron : this->neurons)
// //         {
// //             out.push_back(neuron(inputs));
// //         }
// //         return out;
// //     }

// //     std::vector<Value> get_params()
// //     {
// //         std::vector<Value> out;
// //         for (auto neuron : this->neurons)
// //         {
// //             for (auto value : neuron.get_params())
// //             {
// //                 out.push_back(value);
// //             }
// //         }
// //         return out;
// //     }
// // };

// // class MLP
// // {
// // public:
// //     int nin;
// //     vector<int> nouts;
// //     vector<Layer> layers;
// //     MLP(int nin, vector<int> nouts)
// //     {
// //         this->nin = nin;
// //         this->nouts = nouts;
// //         layers.push_back(Layer(nin, nouts[0]));
// //         for (int i = 0; i < nouts.size() - 1; i++)
// //         {
// //             this->layers.push_back(Layer(nouts[i], nouts[i + 1]));
// //         }
// //     }
// //     std::vector<Value> operator()(std::vector<Value> &inputs)
// //     {
// //         std::vector<Value> out = inputs;
// //         for (auto layer : this->layers)
// //         {
// //             out = layer(out);
// //         }
// //         return out;
// //     }
// //     std::vector<Value> get_params()
// //     {
// //         std::vector<Value> out;
// //         for (auto layer : this->layers)
// //         {
// //             for (auto value : layer.get_params())
// //             {
// //                 out.push_back(value);
// //             }
// //         }
// //         return out;
// //     }
// // };
// // void test_mlp_small()
// // {
// //     MLP mlp = MLP(3, {4, 4, 1});
// //     Layer layer = Layer(2, 3);
// //     vector<double> x = {2.0, 3.0, -1.0};

// //     cout << "Created MLP" << endl;
// //     vector<Value> xv = {Value(2.0), Value(3.0), Value(-1.0)};
// //     // vector<Value> xv = {Value(2.0), Value(3.0), Value(-1.0)};
// //     // Neuron(3)(xv);
// //     vector<Value> out = mlp(xv);
// //     out.back().backward();
// //     cout << "o: " << out.back() << endl;
// //     cout << "o: " << out.back().grad << endl;
// // }
// // void test_mlp_large()
// // {
// //     MLP mlp = MLP(3, {4, 4, 1});

// //     vector<vector<double>> xs = {
// //         {2.0, 3.0, -1.0},
// //         {3.0, -1.0, 0.5},
// //         {0.5, 1.0, 1.0},
// //         {1.0, 1.0, -1.0}};
// //     vector<double> ys = {1.0, -1.0, -1.0, 1.0};
// //     vector<Value> ysv;
// //     for (auto y : ys)
// //     {
// //         ysv.push_back(Value(y));
// //     }
// //     vector<vector<Value>> xsv;
// //     for (auto x : xs)
// //     {
// //         vector<Value> xv;
// //         for (auto xi : x)
// //         {
// //             xv.push_back(Value(xi));
// //         }
// //         xsv.push_back(xv);
// //     }

// //     int num_epochs = 20;
// //     float lr = 10000000;
// //     for (int i = 0; i < num_epochs; i++)
// //     {
// //         //forward pass for all data points

// //         vector<Value> ypred;
// //         for (auto x : xsv)
// //         {
// //             ypred.push_back(mlp(x).back());
// //         }
// //         // compute loss for all data points
// //         Value loss(0.0);
// //         for (int j = 0; j < ysv.size(); j++)
// //         {
// //             Value error = (ysv[j] - ypred[j]).pow(2.0);
// //             loss += error;
// //         }

// //         cout << "Epoch: " << i << " Loss: " << loss << endl;
// //         for (auto param : mlp.get_params())
// //         {
// //             param.grad = 0.0;
// //         }
// //         loss.backward();
// //         for (auto param : mlp.get_params())
// //         {
// //             param.data += -lr * param.grad;
// //         }
// //     }
// // }
// // void test_shared_ptr_by_ref(Value &x)
// // {
// //     cout<<"x: "<<&x<<endl;
// //     cout << "x: " << x << endl;
    
// //     x.data = 10.0;
// //     x.grad = 20.0;
// // }

void test_neuron(){
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
    // Neuron neuron = Neuron(3);
    // shared_ptr<Value> x1 = std::make_shared<Value>(2.0);
    // shared_ptr<Value> x2 = std::make_shared<Value>(3.0);
    // shared_ptr<Value> x3 = std::make_shared<Value>(-1.0);
    // vector<shared_ptr<Value>> x = {x1, x2, x3};
    // auto out = neuron(x);
    // out->backward();
    // cout<<"out: "<<out<<endl;
    // cout<<"out: "<<out->get_grad()<<endl;
}
int main()
{
    // Value x1(2.0), x2(0.0);
    // x1.label = "x1";
    // x2.label = "x2";
    // shared_ptr<Value> x1s = make_shared<Value>(x1);
    // shared_ptr<Value> x2s = make_shared<Value>(x2) ;
    // // vector<Value> xv = {x1s, x2s};
    // cout<<"out:x: "<<&x1s<<endl;
    // cout<<"out:x: " << x1s << endl;
    // test_shared_ptr_by_ref(*x1s);
    // cout<<"out:x: " << *x1s << endl;
    // test_mlp_large();
    test_neuron();
    return 0;
};