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
 * Output shape checking is done in the constructor of the op.
 * 
 * @param[in/out] c Context to use for inference
 * @return Status object stating whether the shape inference was successful.
 */
Status UnpoolShapeFn(InferenceContext* c)
{
    //check input shape has 4 dimensions 
    ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    //check indices has 4 dimensions (bach, width, height, channels)
    ShapeHandle inds_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_shape));

    //check that input dims match
    for(int i = 0; i < 4; i++)
    {
        DimensionHandle input_dim = c->Dim(input_shape,i);
        DimensionHandle ind_dim = c->Dim(input_shape,i);

        if (c->Value(input_dim) != c->Value(ind_dim))
            return errors::InvalidArgument(
                "Indices dimensions must match input dimensions");
    }
    
    std::vector<int32> output_size;
    TF_RETURN_IF_ERROR(c->GetAttr("output_size", &output_size));

    if(output_size.size() != 2)
        return errors::InvalidArgument("'output_size' must be of length 2.");

    //nhwc is the only one supported
    auto output_shape = c->MakeShape({c->Dim(input_shape,0),output_size[0],
            output_size[1],c->Dim(input_shape,3)});

    c->set_output(0, output_shape);

    return Status::OK();
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("Unpool")
        .Input("input: T")
        .Input("inds: Tinds")
        .Output("output: T")
        .Attr("output_size: list(int) >= 2")
        .Attr(GetConvnetDataFormatAttrString())
        .Attr("T: realnumbertype")
        .Attr("Tinds: {int32, int64}")
        .SetShapeFn(UnpoolShapeFn);

//declare kernel launcher
template<typename dtype>
void UnpoolKernelLauncher(const dtype* in, const int64* ind, dtype* out,
                    const int out_N, const int in_N, const int batch,
                    const int height, const int width, const int channels,
                    const int unpooled_h, const int unpooled_w);

/**
 * @brief Unpool operation class.
 * @details This class is a tensorflow operation for unpooling.
 * It takes a tensor and a set of indices and computes the unpooled tensor
 * based on the kernel size and strides.
 * 
 * @tparam dtype the data type for the input to the operation. Either float32
 * or float64.
 */
template <typename dtype>
class UnpoolOp : public OpKernel {
public:
    /**
     * @brief constructor for the unpool operation.
     * @details preform validation on the attributes passed to this operation.
     * This class is analogous to
     * <a href="https://github.com/tensorflow/tensorflow/
     * blob/master/tensorflow/core/kernels/maxpooling_op.cc">
     * maxpooling_op.cc</a>
     * 
     * @param[in/out] context parameter that contains the data format, kernel size,
     * strides, and padding.
     */
    explicit UnpoolOp(OpKernelConstruction* context) 
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

        //check stride has 2 elemenets
        OP_REQUIRES_OK(context, context->GetAttr("output_size", &output_size_));
        OP_REQUIRES(context, output_size_.size() == 2,
                errors::InvalidArgument("'output_size' "
                                        "must be 2 dimensions"));
    }

    /**
     * @brief Perform the actual computation of the upooling operation
     * @details Extract the input tensors, allocate space for the output,
     * and call the kernel launcher.
     * 
     * @param[in/out] context Object that contains input tensors and methods to
     * allocate output
     */
    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& ind_tensor = context->input(1);
        
        //flatten tensors
        auto input = input_tensor.flat<dtype>();
        auto ind = ind_tensor.flat<int64>();

        //UnpoolParameters won't throw any errors but it will modify the
        //state of the context so we need to check if it is still ok to proceed
        UnpoolParameters params{context, output_size_,
                            data_format_, input_tensor.shape(),
                            ind_tensor.shape()};

        if (!context->status().ok()) {
            return;
        }

        // Create an output tensor
        TensorShape out_shape = params.forward_output_shape();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context,
            context->allocate_output(0,
            out_shape,&output_tensor));

        //get flat version to fill
        auto output = output_tensor->flat<dtype>();

        const int out_N = output.size();
        const int in_N = input.size();

        // Call the cuda kernel launcher
        UnpoolKernelLauncher<dtype>(input.data(), ind.data(), output.data(),
                    out_N, in_N, params.tensor_in_batch, params.tensor_in_rows,
                    params.tensor_in_cols, params.depth, params.out_height,
                    params.out_width);
    }

private:
    std::vector<int32> output_size_;
    TensorFormat data_format_;
};


//register kernel with types needed
//maxpool only uses int64
#define REGISTER_KERNEL(type) \
    REGISTER_KERNEL_BUILDER( \
        Name("Unpool") \
        .Device(DEVICE_GPU) \
        .TypeConstraint<type>("T") \
        .TypeConstraint<int64>("Tinds"), \
        UnpoolOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
