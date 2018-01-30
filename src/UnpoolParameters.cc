
#include "UnpoolParameters.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

/**
 * @brief Determine all of the sizes for unpooling, with error checking
 * @details Get all of the shape properties based on the format, and
 * calculate the output shape
 * 
 * @param[in] context context used for generating errors
 * @param[in] stride stride for the unpool operation (needs [1,m,n,1])
 * @param[in] data_format Format of the input data. (Only supports NHWC)
 * @param[in] tensor_in_shape input tensor shape
 * @param[in] tensor_ind_shape indices tensor shape
 */
UnpoolParameters::UnpoolParameters(OpKernelContext* context,
                                    const std::vector<int32>& output_size,
                                    TensorFormat data_format,
                                    const TensorShape& tensor_in_shape,
                                    const TensorShape& tensor_ind_shape) {
    
    std::vector<int32> temp_size;

    if(output_size.size() == 2)
        temp_size.push_back(0); //dummy entry for GetTensorDim
    
    
    for(int i = 0; i < output_size.size(); i++)
    {
        temp_size.push_back(output_size[i]);
    }

    if(output_size.size() == 2)
        temp_size.push_back(0); //another dummy entry

    initialize(context, temp_size, data_format, tensor_in_shape, tensor_ind_shape);
}

UnpoolParameters::UnpoolParameters(OpKernelContext* context,
                                    const TensorShape& output_shape,
                                    TensorFormat data_format,
                                    const TensorShape& tensor_in_shape,
                                    const TensorShape& tensor_ind_shape) {
    std::vector<int32> output_size;
    
    output_size.push_back(0); //dummy entry for GetTensorDim
    output_size.push_back(GetTensorDim(output_shape, data_format, 'H'));
    output_size.push_back(GetTensorDim(output_shape, data_format, 'W'));
    output_size.push_back(0); //another dummy for GetTensorDim

    initialize(context, output_size, data_format, tensor_in_shape, tensor_ind_shape);
}

void UnpoolParameters::initialize(OpKernelContext* context,
                                    const std::vector<int32>& output_size,
                                    TensorFormat data_format,
                                    const TensorShape& tensor_in_shape,
                                    const TensorShape& tensor_ind_shape) {
    // For unpooling, tensor_in should have 4 dimensions.
    OP_REQUIRES(context, tensor_in_shape.dims() == 4,
                            errors::InvalidArgument(
                                "tensor_in must be 4-dimensional"));

    //get all of the dimensions from the input based on the format
    this->data_format = data_format;
    depth = GetTensorDim(tensor_in_shape, data_format, 'C');
    tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
    tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
    tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
    tensor_ind_cols = GetTensorDim(tensor_ind_shape, data_format, 'W');
    tensor_ind_rows = GetTensorDim(tensor_ind_shape, data_format, 'H');
    tensor_ind_batch = GetTensorDim(tensor_ind_shape, data_format, 'N');
    out_height = GetTensorDim(output_size, data_format, 'H');
    out_width = GetTensorDim(output_size, data_format, 'W');
    out_depth = depth;

    //make sure the depth for the indices are the same
    OP_REQUIRES(context, depth == GetTensorDim(tensor_ind_shape, data_format, 'C'),
        errors::InvalidArgument("input depth and indeces depth must match"));
}

/**
 * @brief get the shape of the output in the forward direction
 * @return the shape of the output for the forward direction
 */
TensorShape UnpoolParameters::forward_output_shape() {
        return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                                                     depth);
}
