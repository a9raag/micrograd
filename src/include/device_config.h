#pragma once
#ifndef DEVICE_CONFIG_H
#define DEVICE_CONFIG_H

#include <cstdlib>
#include <string>

namespace device_config {

enum class Device { CUDA, CPU };

inline Device get_device() {
    static const Device device = []() {
        const char* env = std::getenv("MICROGRAD_DEVICE");
        if (env && std::string(env) == "cpu") {
            return Device::CPU;
        }
        return Device::CUDA;
    }();
    return device;
}

inline bool use_cpu() {
    return get_device() == Device::CPU;
}

} // namespace device_config

#endif // DEVICE_CONFIG_H
