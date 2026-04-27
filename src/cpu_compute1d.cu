#include "include/cpu_compute1d.h"
#include "cpu_compute1d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

using namespace std;

template <typename T>
CpuCompute1D<T>::CpuCompute1D() : data(nullptr), size(0) {
    this->shape[0] = 0;
}

template <typename T>
CpuCompute1D<T>::~CpuCompute1D() {
    delete[] data;
}

template <typename T>
CpuCompute1D<T>::CpuCompute1D(size_t size) : size(size) {
    this->shape[0] = size;
    this->data = new T[size]();
}

template <typename T>
T* CpuCompute1D<T>::getData() {
    return this->data;
}

template <typename T>
void CpuCompute1D<T>::setData(T* src) {
    memcpy(this->data, src, this->size * sizeof(T));
}

template <typename T>
T* CpuCompute1D<T>::add(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] + b[i];
    return result;
}

template <typename T>
T* CpuCompute1D<T>::add(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] + static_cast<T>(b);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::dot(BaseCompute<T>& other) {
    T* result = new T[1]();
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[0] += data[i] * b[i];
    return result;
}

template <typename T>
T* CpuCompute1D<T>::mul(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] * b[i];
    return result;
}

template <typename T>
T* CpuCompute1D<T>::mul(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] * static_cast<T>(b);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::greater(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] > b[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::greater(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] > static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::less(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] < b[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::less(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] < static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::equal(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] == b[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::equal(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] == static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::greaterEqual(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] >= b[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::greaterEqual(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] >= static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::lessEqual(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] <= b[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::lessEqual(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] <= static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::pow(float n) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::pow(static_cast<float>(data[i]), n));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::tanh() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::tanh(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::log() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::log(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::exp() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::exp(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::sigmoid() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(1.0f / (1.0f + std::exp(-static_cast<float>(data[i]))));
    return result;
}

template <typename T>
T* CpuCompute1D<T>::relu() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] > static_cast<T>(0) ? data[i] : static_cast<T>(0);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::sum() {
    T* result = new T[1]();
    for (size_t i = 0; i < size; ++i)
        result[0] += data[i];
    return result;
}

template <typename T>
T* CpuCompute1D<T>::sum(int axis) {
    if (axis != 0)
        throw invalid_argument("Invalid axis for 1D array. Only axis=0 is allowed.");
    return sum();
}

template <typename T>
T* CpuCompute1D<T>::subArray(vector<vector<size_t>> dimRanges) {
    if (dimRanges.size() != 1)
        throw invalid_argument("Invalid dimRanges for 1D array.");
    if (dimRanges[0].size() != 2)
        throw invalid_argument("Start and end index must be provided for 1D array.");
    size_t start = dimRanges[0][0];
    size_t end = dimRanges[0][1];
    size_t newSize = end - start;
    T* result = new T[newSize];
    memcpy(result, data + start, newSize * sizeof(T));
    return result;
}

template <typename T>
void CpuCompute1D<T>::fill(T val) {
    for (size_t i = 0; i < size; ++i)
        data[i] = val;
}

template <typename T>
void CpuCompute1D<T>::fillRandom(unsigned int seed) {
    mt19937 rng(seed);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<T>(dist(rng));
}

template <typename T>
int* CpuCompute1D<T>::toInt() {
    int* result = new int[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<int>(data[i]);
    return result;
}

template <typename T>
float* CpuCompute1D<T>::toFloat() {
    float* result = new float[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<float>(data[i]);
    return result;
}

template <typename T>
T* CpuCompute1D<T>::fancyIndexing(vector<vector<size_t>> indices) {
    if (indices.size() != 1)
        throw invalid_argument("Indices must have exactly 1 dimension for 1D array.");
    const vector<size_t>& idx = indices[0];
    size_t newSize = idx.size();
    T* result = new T[newSize];
    for (size_t i = 0; i < newSize; ++i)
        result[i] = data[idx[i]];
    return result;
}
