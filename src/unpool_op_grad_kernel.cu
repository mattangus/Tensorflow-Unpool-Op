

//Uncomment the next line when integrating with tf
//#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/platform/default/integral_types.h"

using namespace tensorflow;
/**
 * @brief CUDA Kernel for computing the gradient of the unpool op
 * @details Just assigns the gradient input to the gradient output
 * based on the indices
 * 
 * @param[in] in input tensor for unpool
 * @param[in] ind indices used to unpool
 * @param[in] grad input gradient from lower layers
 * @param[out] grad_out output gradient for this layer
 * @param[in] nthreads number of threads created for computation
 * @param[in] grad_N number of elements in the gradient tensor
 * @param[in] batch number of elements in the batch
 * @param[in] height height of each image
 * @param[in] width width of each image
 * @param[in] channels number of channels in each image
 * @param[in] unpooled_h the unpooled height (output height)
 * @param[in] unpooled_w the unpooled width (output width)
 * @tparam data type for the input, gradient and output gradient tensors
 */
template <typename dtype>
__global__ void UnpoolGradKernel(const int64* ind,
                            const dtype* grad, dtype* grad_out,
                            const int nthreads,
                            const int grad_N, const int batch, 
                            const int height, const int width,
                            const int channels, const int unpooled_h,
                            const int unpooled_w) {

    //loop over the allocated block for this thread
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        //calculate the which batch the index belongs to 
        int b = (index / (channels * width * height)) % batch;

        //index for the height, width and channel is stored in ind
        //we only need to add the batch to it (with output height and width)
        int new_index = b*channels*unpooled_w*unpooled_h + ind[index];
        //check that the index isn't out of bounds
        if(new_index < grad_N)
            grad_out[index] = grad[new_index];
    }
}

/**
 * @brief Function that launches the unpool gradient kernel
 * @details Zero out gradout and grad_ind_out before launching
 * the unpool gradient kernel
 * 
 * @param[in] in input tensor for unpool
 * @param[in] ind indices used to unpool
 * @param[in] grad input gradient from lower layers
 * @param[out] grad_out output gradient for this layer
 * @param[out] grad_ind_out output gradient for the indices (just zeroed out)
 * @param[in] g_ind_N number of elements in the gradient for the indices
 * @param[in] grad_N number of elements in the gradient tensor
 * @param[in] batch number of elements in the batch
 * @param[in] height height of each image
 * @param[in] width width of each image
 * @param[in] channels number of channels in each image
 * @param[in] unpooled_h the unpooled height (output height)
 * @param[in] unpooled_w the unpooled width (output width)
 * @tparam data type for the input, gradient and output gradient tensors
 */
template <typename dtype>
void UnpoolGradKernelLauncher(const int64* ind,
                            const dtype* grad, dtype* grad_out,
                            dtype* grad_ind_out, const int grad_out_N,
                            const int g_ind_N, const int grad_N,
                            const int batch, const int height,
                            const int width, const int channels,
                            const int unpooled_h, const int unpooled_w)
{
    const int kThreadsPerBlock = 1024;

    //zero out grad and inds grad
    SetZero<<<(grad_out_N + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(grad_out_N, grad_out);

    SetZero<<<(g_ind_N + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(g_ind_N, grad_ind_out);

    //fill grad based on indices
    UnpoolGradKernel<dtype><<<(grad_out_N + kThreadsPerBlock - 1) / kThreadsPerBlock,
                    kThreadsPerBlock>>>(ind, grad, grad_out,
                        grad_out_N, grad_N,
                        batch, height, width, channels,
                        unpooled_h, unpooled_w);

    //wait for computation to complete and check for errors
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(cudaerr));
}


//forward declaration for all the types needed
#define UNPOOL_KERNEL_GRAD_TYPE(type)                   \
    template void UnpoolGradKernelLauncher<type>(       \
        const int64* ind,               \
        const type* grad, type* grad_out,               \
        type* grad_ind_out, const int grad_out_N,       \
        const int g_ind_N, const int grad_N,            \
        const int batch, const int height,              \
        const int width, const int channels,            \
        const int unpooled_h, const int unpooled_w)

UNPOOL_KERNEL_GRAD_TYPE(float);
UNPOOL_KERNEL_GRAD_TYPE(double);

#undef UNPOOL_KERNEL_GRAD_TYPE
//Uncomment the next line when integrating with tf source
//#endif
