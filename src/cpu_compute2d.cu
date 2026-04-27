#include "include/cpu_compute2d.h"
#include "include/cpu_compute1d.h"
#include "cpu_compute2d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace std;

enum CPU_MATRIX_TYPE { CPU_ROW_WISE, CPU_COL_WISE, CPU_NORMAL, CPU_SINGLETON };

template <typename T>
static CPU_MATRIX_TYPE cpuMatrixType(BaseCompute<T>& lhs, BaseCompute<T>& rhs) {
    if (rhs.getSize() == 1) return CPU_SINGLETON;

    if (rhs.getShape()[0] == 1 && rhs.getShape()[1] == lhs.getShape()[1])
        return CPU_ROW_WISE;

    if (rhs.getShape()[0] == lhs.getShape()[0] && rhs.getShape()[1] == 1)
        return CPU_COL_WISE;

    if (dynamic_cast<CpuCompute1D<T>*>(&rhs) != nullptr &&
        lhs.getShape()[1] == rhs.getSize())
        return CPU_ROW_WISE;

    if (rhs.getShape()[0] == lhs.getShape()[0] && rhs.getShape()[1] == lhs.getShape()[1])
        return CPU_NORMAL;

    ostringstream error;
    error << "Shape mismatch: " << lhs.getShape()[0] << "x" << lhs.getShape()[1]
          << " vs " << rhs.getShape()[0] << "x" << rhs.getShape()[1];
    throw invalid_argument(error.str());
}

template <typename T>
CpuCompute2D<T>::CpuCompute2D() : data(nullptr), size(0) {
    this->shape[0] = 0;
    this->shape[1] = 0;
}

template <typename T>
CpuCompute2D<T>::~CpuCompute2D() {
    delete[] data;
}

template <typename T>
CpuCompute2D<T>::CpuCompute2D(size_t x, size_t y) {
    this->shape[0] = x;
    this->shape[1] = y;
    this->size = x * y;
    this->data = new T[size]();
}

template <typename T>
T* CpuCompute2D<T>::getData() {
    return this->data;
}

template <typename T>
void CpuCompute2D<T>::setData(T* src) {
    memcpy(this->data, src, this->size * sizeof(T));
}

template <typename T>
T* CpuCompute2D<T>::transpose() {
    size_t x = shape[0], y = shape[1];
    T* result = new T[x * y];
    for (size_t i = 0; i < x; ++i)
        for (size_t j = 0; j < y; ++j)
            result[j * x + i] = data[i * y + j];
    return result;
}

template <typename T>
T* CpuCompute2D<T>::add(BaseCompute<T>& other) {
    size_t x = shape[0], y = shape[1];
    T* result = new T[size];
    T* b = other.getData();
    CPU_MATRIX_TYPE type = cpuMatrixType(*this, other);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            T rhs;
            switch (type) {
                case CPU_SINGLETON:  rhs = b[0]; break;
                case CPU_ROW_WISE:   rhs = b[j]; break;
                case CPU_COL_WISE:   rhs = b[i]; break;
                default:             rhs = b[i * y + j]; break;
            }
            result[i * y + j] = data[i * y + j] + rhs;
        }
    }
    return result;
}

template <typename T>
T* CpuCompute2D<T>::add(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] + static_cast<T>(b);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::mul(BaseCompute<T>& other) {
    size_t x = shape[0], y = shape[1];
    T* result = new T[size];
    T* b = other.getData();
    CPU_MATRIX_TYPE type = cpuMatrixType(*this, other);
    for (size_t i = 0; i < x; ++i) {
        for (size_t j = 0; j < y; ++j) {
            T rhs;
            switch (type) {
                case CPU_SINGLETON:  rhs = b[0]; break;
                case CPU_ROW_WISE:   rhs = b[j]; break;
                case CPU_COL_WISE:   rhs = b[i]; break;
                default:             rhs = b[i * y + j]; break;
            }
            result[i * y + j] = data[i * y + j] * rhs;
        }
    }
    return result;
}

template <typename T>
T* CpuCompute2D<T>::mul(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] * static_cast<T>(b);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::dot(BaseCompute<T>& other) {
    size_t heightA = shape[0], widthA = shape[1];
    size_t widthB = other.getShape()[1];
    if (widthA != other.getShape()[0]) {
        ostringstream err;
        err << "dot: width " << widthA << " != height " << other.getShape()[0];
        throw invalid_argument(err.str());
    }
    T* b = other.getData();
    T* result = new T[heightA * widthB]();
    for (size_t row = 0; row < heightA; ++row)
        for (size_t col = 0; col < widthB; ++col)
            for (size_t k = 0; k < widthA; ++k)
                result[row * widthB + col] += data[row * widthA + k] * b[k * widthB + col];
    return result;
}

template <typename T>
T* CpuCompute2D<T>::greater(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] > b[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::greater(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] > static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::less(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] < b[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::less(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] < static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::equal(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] == b[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::equal(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] == static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::greaterEqual(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] >= b[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::greaterEqual(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] >= static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::lessEqual(BaseCompute<T>& other) {
    T* result = new T[size];
    T* b = other.getData();
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] <= b[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::lessEqual(float b) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(data[i] <= static_cast<T>(b));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::pow(float n) {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::pow(static_cast<float>(data[i]), n));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::tanh() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::tanh(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::log() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::log(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::exp() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(std::exp(static_cast<float>(data[i])));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::sigmoid() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<T>(1.0f / (1.0f + std::exp(-static_cast<float>(data[i]))));
    return result;
}

template <typename T>
T* CpuCompute2D<T>::relu() {
    T* result = new T[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = data[i] > static_cast<T>(0) ? data[i] : static_cast<T>(0);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::sum() {
    T* result = new T[1]();
    for (size_t i = 0; i < size; ++i)
        result[0] += data[i];
    return result;
}

template <typename T>
T* CpuCompute2D<T>::sum(int axis) {
    size_t x = shape[0], y = shape[1];
    if (axis == 0) {
        T* result = new T[y]();
        for (size_t i = 0; i < x; ++i)
            for (size_t j = 0; j < y; ++j)
                result[j] += data[i * y + j];
        return result;
    } else if (axis == 1) {
        T* result = new T[x]();
        for (size_t i = 0; i < x; ++i)
            for (size_t j = 0; j < y; ++j)
                result[i] += data[i * y + j];
        return result;
    } else {
        throw invalid_argument("Axis must be 0 or 1");
    }
}

template <typename T>
T* CpuCompute2D<T>::subArray(vector<vector<size_t>> dimRanges) {
    if (dimRanges.size() != 2)
        throw invalid_argument("dimRanges must have 2 dimensions");
    size_t startX = dimRanges[0][0], endX = dimRanges[0][1];
    size_t startY = dimRanges[1][0], endY = dimRanges[1][1];
    size_t rx = endX - startX, ry = endY - startY;
    T* result = new T[rx * ry];
    for (size_t i = 0; i < rx; ++i)
        for (size_t j = 0; j < ry; ++j)
            result[i * ry + j] = data[(i + startX) * shape[1] + (j + startY)];
    return result;
}

template <typename T>
void CpuCompute2D<T>::fill(T val) {
    for (size_t i = 0; i < size; ++i)
        data[i] = val;
}

template <typename T>
void CpuCompute2D<T>::fillRandom(unsigned int seed) {
    mt19937 rng(seed);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<T>(dist(rng));
}

template <typename T>
int* CpuCompute2D<T>::toInt() {
    int* result = new int[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<int>(data[i]);
    return result;
}

template <typename T>
float* CpuCompute2D<T>::toFloat() {
    float* result = new float[size];
    for (size_t i = 0; i < size; ++i)
        result[i] = static_cast<float>(data[i]);
    return result;
}

template <typename T>
T* CpuCompute2D<T>::fancyIndexing(vector<vector<size_t>> indices) {
    if (indices.size() != 2)
        throw invalid_argument("Indices must be a 2D array");
    if (indices[0].size() != 1 || indices[1].size() != 1)
        throw invalid_argument("Each dimension range must be a 1D array of size 1");
    size_t ix = indices[0][0], iy = indices[1][0];
    T* result = new T[1];
    result[0] = data[ix * shape[1] + iy];
    return result;
}
