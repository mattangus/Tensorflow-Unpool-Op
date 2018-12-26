cmake_minimum_required(VERSION 3.5)

#get the include directory for tensorflow
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include(), end='')" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)
execute_process(COMMAND python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end='')" OUTPUT_VARIABLE Tensorflow_LIB_DIRS)
 
message("tensorflow include dir: ${Tensorflow_INCLUDE_DIRS}")
message("tensorflow link dir: ${Tensorflow_LIB_DIRS}")
include_directories(include)
include_directories(${Tensorflow_INCLUDE_DIRS})
include_directories("/usr/local/")
link_directories(${Tensorflow_LIB_DIRS})

find_package(CUDA)

#set flags based on tutorial
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11  -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr" )

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

target_link_libraries(unpool_op "tensorflow_framework")

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