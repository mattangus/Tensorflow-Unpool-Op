#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace shape_inference;

/**
 * @brief Class used to compute and store information about the unpool op
 * @details Calculate and store information such as size of input and output.
 * This is done based on the stride, kernel size, input size and format.
 */
class UnpoolParameters {
public:
    // Updates context->status if there is an invalid input.
    UnpoolParameters(OpKernelContext* context, const std::vector<int32>& output_size,
                    TensorFormat data_format, const TensorShape& tensor_in_shape,
                    const TensorShape& tensor_ind_shape);

    // Second constructor for convenience.
    UnpoolParameters(OpKernelContext* context, const TensorShape& output_shape,
                    TensorFormat data_format, const TensorShape& tensor_in_shape,
                    const TensorShape& tensor_ind_shape);

    // Returns the shape of the output for "forward" pooling operations.
    TensorShape forward_output_shape();

    int depth;

    int tensor_in_cols;
    int tensor_in_rows;
    int tensor_in_batch;

    int tensor_ind_cols;
    int tensor_ind_rows;
    int tensor_ind_batch;

    int64 out_height;
    int64 out_width;
    int out_depth;

    TensorFormat data_format;
private:
    void initialize(OpKernelContext* context, const std::vector<int32>& output_size,
                    TensorFormat data_format, const TensorShape& tensor_in_shape,
                    const TensorShape& tensor_ind_shape);
};
