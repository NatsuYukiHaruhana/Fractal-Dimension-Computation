#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t CudaDBC2D(unsigned char* I, const int M, const unsigned char G, float Nr[]);