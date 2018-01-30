cmake_minimum_required(VERSION 3.5)

#get the include directory for tensorflow
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories(include)

find_package(CUDA)

#set flags based on tutorial
set (CMAKE_CXX_FLAGS "--std=c++11 -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1")

set(CMAKE_BUILD_TYPE Debug)

#pass flags to c++ compiler
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

#create library
cuda_add_library(
	unpool_op SHARED
	src/unpool_op_kernel.cu
	src/unpool_op_kernel.cc
	src/unpool_op_grad_kernel.cc
	src/unpool_op_grad_kernel.cu
	src/UnpoolParameters.cc)

#copy python files to build folder (for testing)
file(GLOB PY_FILES
  "src/*.py"
)
file(COPY ${PY_FILES} DESTINATION .)

set(CMAKE_INSTALL_PREFIX ".")

set(RELATIVE_PATH "../tf_rewrite/segnet/")

#install library file
install(TARGETS unpool_op
	DESTINATION ${RELATIVE_PATH})

#install python interface
install(FILES "src/unpool.py"
	DESTINATION ${RELATIVE_PATH})