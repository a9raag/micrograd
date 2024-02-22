#include <iostream> 

using namespace std;
template <typename T>
class BaseCompute{ 
    public: 
        virtual T* add(T* b, size_t* shape, size_t size) = 0; 
        virtual T* add(double b, size_t* shape, size_t size) = 0;
        virtual T* mul(T* b, size_t* shape, size_t size) = 0;
        virtual T* mul(double b) = 0;
        virtual T* dot(T* b, size_t* shape, size_t size) = 0;
        // virtual T* dot(double b, size_t* shape, size_t size) = 0;
        virtual T* neg() = 0;
        virtual T* pow(double b) = 0;
        virtual T* tanh() = 0;
        virtual void fill(T val) = 0;
        virtual T* getData() = 0;
        virtual void setData(T *data) = 0; 
};
