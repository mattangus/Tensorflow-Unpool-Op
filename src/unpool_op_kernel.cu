/**
 * The functions in this file are analogs to
 * <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/maxpooling_op_gpu.cu.cc">
 * maxpooling_op_gpu.cu.cc</a>
 */

//Uncomment the next line when integrating with tf
//#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

/**
 * @brief CUDA Kernel for computing the unpool op
 * @details Just assigns the input to the output based on the indices
 * 
 * @param[in] in input tensor containing values to unpool
 * @param[in] ind indices for where to put input values in the output
 * @param[out] out tensor to store result in
 * @param[in] nthreads number of threads being created for the computation
 * @param[in] out_N size of the output
 * @param[in] batch number of elements in the batch
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] channels number of channels of input and output
 * @param[in] unpooled_h height of the output
 * @param[in] unpooled_w width of the output
 */
template <typename dtype> __global__ void UnpoolKernel(const dtype* in, const int64* ind, dtype* out,
                            const int nthreads, const int out_N,
                            const int batch, const int height,
                            const int width, const int channels,
                            const int unpooled_h, const int unpooled_w) {

    //loop over the allocated block for this thread
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        //calculate the which batch the index belongs to
        int b = (index / (channels * width * height)) % batch;

        //index for the height, width and channel is stored in ind
        //we only need to add the batch to it (with output height and width)
        int new_index = b*channels*unpooled_w*unpooled_h + ind[index];
        //check that the index isn't out of bounds
        if(new_index < out_N)
            out[new_index] = in[index];
    }
}

/**
 * @brief Function that prepares the data and launches the unpool kernel
 * @details Zero out the output tensor before launching the unpool gradient kernel
 * 
 * @param[in] in input tensor containing values to unpool
 * @param[in] ind indices for where to put input values in the output
 * @param[out] out tensor to store result in
 * @param[in] in_N size of the input
 * @param[in] out_N size of the output
 * @param[in] batch number of elements in the batch
 * @param[in] height height of input
 * @param[in] width width of input
 * @param[in] channels number of channels of input and output
 * @param[in] unpooled_h height of the output
 * @param[in] unpooled_w width of the output
 */
template <typename dtype>
void UnpoolKernelLauncher(const dtype* in, const int64* ind, dtype* out,
                        const int out_N, const int in_N, const int batch,
                        const int height, const int width, const int channels,
                        const int unpooled_h, const int unpooled_w)
{
    const int kThreadsPerBlock = 1024;
    
    //zero out the output
    SetZero<<<(out_N + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(out_N, out);

    //fill the output based on the indices
    UnpoolKernel<dtype><<<(in_N + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(
            in, ind, out, in_N, out_N, batch, height, width,
            channels, unpooled_h, unpooled_w);

    //wait for computation to complete, and check for errors
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
}

//forward declaration for all the types needed
#define UNPOOL_KERNEL_TYPE(type)                                \
    template void UnpoolKernelLauncher<type>(                   \
        const type* in, const int64* ind, type* out,            \
        const int out_N, const int in_N, const int batch,       \
        const int height, const int width, const int channels,  \
        const int unpooled_h, const int unpooled_w)

UNPOOL_KERNEL_TYPE(float);
UNPOOL_KERNEL_TYPE(double);

#undef UNPOOL_KERNEL_TYPE

//Uncomment the next line when integrating with tf
//#endif
