#include "sigmod.hpp"
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

__global__ void sigmod_kernel(float *input, float *output, uint length) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
	// uint y = blockIdx.y * blockDim.y  + threadIdx.y;
    if(x >= length) {
        return;
    }

    output[x] = 1.0 / (1.0 + expf(input[x]));
}

void sigmod_layer::forward(float *input, float *output){
    CHECK(m_length != 0) << "m_length is 0";

    float* d_input;
    eee(cudaMalloc((void**)&d_input, sizeof(float) * m_length));
    float* d_output;
    eee(cudaMalloc((void**)&d_output, sizeof(float) * m_length));

    eee(cudaMemcpy(d_input, input, m_length*sizeof(float), cudaMemcpyHostToDevice));

    uint thread_PerBlock = 32;
    dim3 rowsGrid(ceil(1.0f*m_length/thread_PerBlock),1 , 1);
	dim3 rowsThreads(thread_PerBlock, 1, 1);

    sigmod_kernel<<<rowsGrid, rowsThreads>>>(d_input, d_output, m_length);

    eee(cudaMemcpy(output, d_output, m_length * sizeof(float), cudaMemcpyDeviceToHost));

}