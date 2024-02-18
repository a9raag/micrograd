#include <iostream> 
#include "compute.cu"
#include "cuda_compute.cu"
#include <stdexcept>
using namespace std;
template <typename T = double>
class Compute1D : public BaseCompute<T> {
    private: 
        int threadsPerBlock;
        int blocksPerGrid;
        int allocSize;
        long size; 
        T* data;
    public:
        Compute1D() {
        }
        ~Compute1D() {
            cout<<"Freeing memory"<<endl;
            cudaFree(this->data);
        }
        T* getData() {
            return this->data;
        }
        void setData(T* data) {
            this->data = data;
        }
        Compute1D(long size){
            data = new T[size];
            this->size = size;
            this->threadsPerBlock = 256;
            this->blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        
            // Initialize data on GPU
            int allocSize = size * sizeof(T);
            if(cudaMallocManaged(&this->data, allocSize) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
            }
            // debug data 
            // vector<T> cdata(size);
            // if (cudaMemcpy(cdata.data(), this->data, allocSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
            //     cout<<"Error in copying data to GPU"<<endl;
            //     cudaFree(this->data);
            // }
            // cout<<"Data copied to GPU"<<endl;
            // cout<<"Data: ";
            // for(int i = 0; i < size; i++){
            //     cout<<cdata[i]<<" ";
            // }
            // cout<<endl;
            cudaDeviceSynchronize();
        }
        
        Compute1D(vector<T> hdata, int size) {
            if (cudaMemcpy(this->data, hdata.data(), allocSize, cudaMemcpyHostToDevice) != cudaSuccess) {
                cout<<"Error in copying data to GPU"<<endl;
                cudaFree(this->data);
            }
            *this = Compute1D(size);
        }

        T* add(T* b) {
            T* c;  
            if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
            cudaDeviceSynchronize();
            return c;
        }
        
        T* add(double b) {
            T* c;  
            if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            addKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
            cudaDeviceSynchronize();
            return c;
        }

        T* dot(T * b){ 
            T *c; 
            
            if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
            cudaDeviceSynchronize();
            return c;

        }
        T* dot(double b){ 
            T *c; 
            
            if(cudaMallocManaged(&c, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, b, c, size);
            cudaDeviceSynchronize();
            return c;

        }
    
        // convert matrix vals to negative 
        T* neg(){ 
            T* out; 
            if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            dotKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, -1.0, out, size);
            cudaDeviceSynchronize();
            return out;
        }

        T* pow(double n){ 
            T* out; 
            if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            powKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, n, size);
            cudaDeviceSynchronize();
            return out;
        }

        T* tanh(){ 
            T* out; 
            if(cudaMallocManaged(&out, size * sizeof(T)) != cudaSuccess){
                cout<<"Error in allocating memory"<<endl;
                throw runtime_error("Error in allocating memory");
            }
            tanhKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, out, size);

            cudaDeviceSynchronize();
            return out;
        }

        void fill(T val){
            fillKernel<<<blocksPerGrid, threadsPerBlock>>>(this->data, val, size);
            cudaDeviceSynchronize();
        }
        void toDevice() {
            cudaMemcpy(this->data, this->data, allocSize, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        void toHost() {
            cout<<"Copy to host"<<endl;
            cudaMemcpy(this->data, this->data, allocSize, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
};