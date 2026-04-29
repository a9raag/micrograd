#pragma once
#ifndef HELPER_H
#define HELPER_H

#include "engine.h"

template <typename T>
Tensor<T> oneHot(Tensor<T> &input, int num_classes);

#endif