#include "include/helper.h"


template <typename T> 
Tensor<T> oneHot(Tensor<T> &input, size_t num_classes){
    Tensor<T> result = Tensor<T>({input.getShape()[0], num_classes});
    for (int i = 0; i < input.getShape()[0]; i++){
        result(i, (int) input(i)) = 1;
    }
    return result;
}