#include <iostream> 

using namespace std;
template <typename T>
class BaseCompute{ 
    public: 
        virtual T* add(T* b) = 0; 
        virtual T* add(double b) = 0;
        virtual T* neg() = 0;
        virtual T* dot(T* b) = 0;
        virtual T* dot(double b) = 0;
        virtual T* pow(double b) = 0;
        virtual T* tanh() = 0;
        virtual void fill(T val) = 0;
        // virtual T* div(T* b) = 0;
        virtual void toHost() = 0;
        virtual T* getData() = 0;
        virtual void setData(T *data) = 0; 
};
