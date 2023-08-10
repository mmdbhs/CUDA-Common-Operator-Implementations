#include "relu.hpp"
// #include "tensor.hpp"
#include <glog/logging.h>


#define eee(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void relu_kernel(float *input, float *output, uint thread_PerBlock, uint length) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y  + threadIdx.y;
    if(y*thread_PerBlock+x >= length) {
        return;
    }

    if(input[y*thread_PerBlock + x] >=0)
    {
        output[y*thread_PerBlock + x] = input[y*thread_PerBlock+x];
    } else {
        output[y*thread_PerBlock + x] = 0;
    }
}

void relu_layer::forward(float *input, float *output){
    CHECK(m_length != 0) << "m_length is 0";

    float* d_input;
    eee(cudaMalloc((void**)&d_input, sizeof(float) * m_length));
    float* d_output;
    eee(cudaMalloc((void**)&d_output, sizeof(float) * m_length));

    eee(cudaMemcpy(d_input, input, m_length*sizeof(float), cudaMemcpyHostToDevice));

    uint thread_PerBlock = 32;
    dim3 rowsGrid(1, ceil(1.0f*m_length/thread_PerBlock), 1);
	dim3 rowsThreads(thread_PerBlock, 1, 1);

    relu_kernel<<<rowsGrid, rowsThreads>>>(d_input, d_output, thread_PerBlock, m_length);

    eee(cudaMemcpy(output, d_output, m_length * sizeof(float), cudaMemcpyDeviceToHost));

}