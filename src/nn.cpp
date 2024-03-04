#include <iostream>
#include <set>
#include <memory>
#include <list>
#include <math.h>
#include <random>
#include <vector>
#include "include/engine.h"
using namespace std;
class Layer
{
private: 
    shared_ptr<Value> weights;
    shared_ptr<Value> bias;
public:
    Layer() {} 
    vector<size_t> n_inputs;
    
    
    Layer(size_t n_input, size_t n_outs)
    {

        // std::random_device rd;  // Will be used to obtain a seed for the random number engine
        // std::mt19937 gen(rd()); // Standard mersenne_twisterx_engine seeded with rd()
        // std::uniform_real_distribution<> dis(-1.0, 1.0);

        this->n_inputs = n_inputs;
        Tensor<double> wt = Tensor<double>({n_input, n_outs}).randomize();
        this->weights = make_shared<Value>(wt);
        
    }
    shared_ptr<Value> get_bias()
    {
        return this->bias;
    }

    shared_ptr<Value> operator()(shared_ptr<Value> &inputs)
    {
        shared_ptr<Value> val = inputs->dot(this->weights);
        auto bias_tensor = Tensor<double>(val->getData().shape);
        bias_tensor.randomize();
        this->bias = make_shared<Value>(bias_tensor);
        return val + bias;
    }

    std::vector<shared_ptr<Value>> get_params()
    {
        std::vector<shared_ptr<Value>> out;
        out.emplace_back(this->weights);
        out.emplace_back(this->bias);
        return out;
    }
};

class MLP{
    private: 
        vector<Layer> layers;
    public:
        MLP(size_t n_input, vector<size_t> n_outs){
            for (size_t i = 0; i < n_outs.size(); i++)
            {
                if(i == 0){
                    layers.emplace_back(Layer(n_input, n_outs[i]));
                }else{
                    layers.emplace_back(Layer(n_outs[i-1], n_outs[i]));
                }
            }
        }
        shared_ptr<Value> operator()(shared_ptr<Value> &inputs){
            shared_ptr<Value> out = inputs;
            for (auto &layer : layers)
            {
                out = layer(out);
            }
            return out;
        }

        vector<shared_ptr<Value>> get_params(){
        vector<shared_ptr<Value>> out;
            for (auto &layer : layers)
            {
                auto params = layer.get_params();
                out.insert(out.end(), params.begin(), params.end());
            }
            return out;
        }

        void update_params(double lr){
            for (auto &layer : layers)
            {
                auto params = layer.get_params();
                for (shared_ptr<Value> &param : params)
                {
                    auto data = param->getData() - param->getGrad() * lr;
                    param->setData(data);
                }
            }
        }
        
        void zero_grad(){
            for (auto &layer : layers)
            {
                auto params = layer.get_params();
                for (shared_ptr<Value> &param : params)
                {
                    param->getGrad().fill(0.0);
                }
            }
        }
};


// void test_mlp_small()
// {
//     MLP mlp = MLP(3, {4, 4, 1});
//     cout << "Created MLP" << endl;
//     vector<shared_ptr<Value>> xv = {
//         make_shared<Value>(2.0), 
//         make_shared<Value>(3.0), 
//         make_shared<Value>(-1.0)
//         };
//     vector<shared_ptr<Value>> out = mlp(xv);
//     out.back()->backward();
//     cout << "o: " << out.back() << endl;
//     cout << "o: " << out.back()->get_grad() << endl;
// }
// void test_mlp_large()
// {
//     MLP mlp = MLP(3, {4, 4, 1});

//     vector<vector<double>> xs = {
//         {2.0, 3.0, -1.0},
//         {3.0, -1.0, 0.5},
//         {0.5, 1.0, 1.0},
//         {1.0, 1.0, -1.0}};
//     vector<double> ys = {1.0, -1.0, -1.0, 1.0};
//     vector<shared_ptr<Value>> ysv;
//     for (auto y : ys)
//     {
//         ysv.emplace_back(make_shared<Value>(y));
//     }
//     vector<vector<shared_ptr<Value>>> xsv;
//     for (auto x : xs)
//     {
//         vector<shared_ptr<Value>> xv;
//         for (auto xi : x)
//         {
//             xv.emplace_back(make_shared<Value>(xi));
//         }
//         xsv.emplace_back(xv);
//     }

//     int num_epochs = 100;
//     float lr = 0.001;
//     for (int i = 0; i < num_epochs; i++)
//     {
//         //forward pass for all data points

//         vector<shared_ptr<Value>> ypred;
//         for (auto x : xsv)
//         {
//             ypred.emplace_back(mlp(x).back());
//         }
//         // compute loss for all data points
//         shared_ptr<Value> loss= make_shared<Value>(0.0);
//         for (int j = 0; j < ysv.size(); j++)
//         {
//             shared_ptr<Value> error = (ysv[j] - ypred[j])->pow(2.0);
//             loss = loss + error;
//         }
//         loss = loss/make_shared<Value>(ysv.size());

//         for (auto param : mlp.get_params())
//         {
//             param->set_grad(0.0);
//         }

//         loss->backward();

//         cout << "Epoch: " << i << " Loss: " << loss << endl;
//         for (auto param : mlp.get_params())
//         {
//             param->set_data( param->get_data()  -lr * param->get_grad());
//         }
//     }
    
//     cout << "Done training" << endl;
// }


// void test_layer(){
//     Layer layer = Layer(3, 4);
//     shared_ptr<Value> x1 = std::make_shared<Value>(2.0);
//     shared_ptr<Value> x2 = std::make_shared<Value>(3.0);
//     shared_ptr<Value> x3 = std::make_shared<Value>(-1.0);
//     vector<shared_ptr<Value>> x = {x1, x2, x3};
//     shared_ptr<Value> y1 = std::make_shared<Value>(3.0);
//     shared_ptr<Value> y2 = std::make_shared<Value>(4.0);
//     shared_ptr<Value> y3 = std::make_shared<Value>(-5.0);
//     cout<<"out: "<<x1<<endl;
//     cout<<"out: "<<x2<<endl;
//     cout<<"out: "<<x3<<endl;
//     auto out = layer(x);
//     cout<<"initate backward on out: "<<out[0]<<endl;
//     out.back()->backward();
//     cout<<"out: "<<out[0]<<endl;
//     cout<<"out: "<<out[0]->get_grad()<<endl;

// }
// int main(int argc, char const *argv[])
// {
//     // test_mlp_large();
//     // test_neuron();
//     // cout<<"[SUCCESS] test neuron"<<endl;
//     // test_layer();
//     // cout<<"[SUCCESS] test layer"<<endl;
//     // test_mlp_small();
//     // cout<<"[SUCCESS] test mlp small"<<endl;
//     test_mlp_large();
//     cout<<"[SUCCESS] test mlp large"<<endl;
//     return 0;
// };