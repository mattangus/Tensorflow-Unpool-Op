#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

#include "UnpoolParameters.h"

#include <iostream>
#include <cuda.h>

using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace shape_inference;

/**
 * @brief Infer the shape of the output based on the input tensors
 * @details Check that the input and indices tensors' shape match.
 * Set the output gradient shape to be the same as the input shape.
 * 
 * @param[in/out] c Context to use for inference
 * @return Status object stating whether the shape inference was successful.
 */
Status UnpoolGradShapeFn(InferenceContext* c)
{
    //check indices has 4 dimensions (bach, width, height, channels)
    ShapeHandle inds_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &inds_shape));

    c->set_output(0, inds_shape);
    c->set_output(1, inds_shape);
    return Status::OK();
}

/**
 * register the gradient operation with necessary options
 */
REGISTER_OP("UnpoolGrad")
        .Input("inds: Tinds")
        .Input("grad: T")
        .Attr(GetConvnetDataFormatAttrString())
        .Attr("T: realnumbertype")
        .Attr("Tinds: {int32, int64}")
        .Output("grad_input: T")
        .Output("grad_inds: T")
        .SetShapeFn(UnpoolGradShapeFn);

//declare kernel launcher
template <typename dtype>
void UnpoolGradKernelLauncher(const int64* ind,
                            const dtype* grad, dtype* grad_out,
                            dtype* grad_ind_out, const int grad_out_N,
                            const int g_ind_N, const int grad_N,
                            const int batch, const int height,
                            const int width, const int channels,
                            const int unpooled_h, const int unpooled_w);

/**
 * @brief Unpool operation gradient class.
 * @details This class is a tensorflow operation for the unpooling gradient.
 * It takes a tensor and a set of indices and computes the gradient for the 
 * unpooled tensor based on the gradient passed from deeper layers
 * 
 * @tparam dtype the data type for the input to the operation. Either float32
 * or float64.
 */
template <typename dtype>
class UnpoolGradOp : public OpKernel {
public:
    /**
     * @brief constructor for the unpool gradient operation.
     * @details preform validation on the attributes passed to this operation.
     * 
     * @param[in/out] context parameter that contains the data format, kernel size,
     * strides, and padding.
     */
    explicit UnpoolGradOp(OpKernelConstruction* context) 
        : OpKernel(context)
    {
        //get data format
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));

        //only nhwc supported
        OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::InvalidArgument("Unpool only supports NHWC ",
                                "on device type ",
                                DeviceTypeString(context->device_type())));

    }

    /**
     * @brief Perform the computation of the upooling gradient operation
     * @details Extract the input tensors, allocate space for the output,
     * and call the kernel launcher.
     * 
     * @param[in/out] context Object that contains input tensors and methods to
     * allocate output
     */
    void Compute(OpKernelContext* context) override {
        //cout << "unpoolgradup compute" << endl;
        // Grab the input tensors
        const Tensor& ind_tensor = context->input(0);
        const Tensor& grad_tensor = context->input(1);

        //the gradient tensor is the shape of the bottom layer,
        //inds tensor is the shape of the top layer
        //use this helper to compute the output shape
        UnpoolParameters params{context, grad_tensor.shape(),
                            data_format_, ind_tensor.shape(),
                            ind_tensor.shape()};

        //UnpoolParameters won't throw any errors but it will modify the
        //state of the context so we need to check if it is still ok to proceed
        if (!context->status().ok()) {
            return;
        }

        //flatten tensors
        auto ind = ind_tensor.flat<int64>();
        auto grad = grad_tensor.flat<dtype>();

        // Create an output tensors
        Tensor* grad_for_input = nullptr;
        OP_REQUIRES_OK(context,
            context->forward_input_or_allocate_output({0}, 0,
            ind_tensor.shape(),&grad_for_input));

        Tensor* grad_for_inds = nullptr;
        OP_REQUIRES_OK(context,
            context->forward_input_or_allocate_output({1}, 1,
            ind_tensor.shape(),&grad_for_inds));

        //get flat versions for filling
        auto grad_out = grad_for_input->flat<dtype>();
        auto grad_inds = grad_for_inds->flat<dtype>();

        //get sizes needed
        const int grad_out_N = grad_out.size();
        const int g_inds_N = grad_inds.size();
        const int grad_N = grad.size();

        // Call the cuda kernel launcher
        UnpoolGradKernelLauncher<dtype>(ind.data(),
                    grad.data(), grad_out.data(),
                    grad_inds.data(), grad_out_N, g_inds_N, grad_N,
                    params.tensor_in_batch, params.tensor_in_rows,
                    params.tensor_in_cols, params.depth, params.out_height,
                    params.out_width);
    }

private:
    
    TensorFormat data_format_;
};

//register kernels with types needed
//maxpool only uses int64
#define REGISTER_KERNEL(type) \
    REGISTER_KERNEL_BUILDER( \
        Name("UnpoolGrad") \
        .Device(DEVICE_GPU) \
        .TypeConstraint<type>("T") \
        .TypeConstraint<int64>("Tinds"), \
        UnpoolGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
