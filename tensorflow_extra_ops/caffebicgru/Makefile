default: all

all: caffecgrugradient caffecgru

python2: caffecgrugradient2 caffecgru2

python3: caffecgrugradient3 caffecgru3

caffecgrugradient: caffecgrugradient3 caffecgrugradient2

caffecgru: caffecgru3 caffecgru2

caffecgru3: caffe_c_g_r_u_step.cc
	$(eval TF_INC := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'))
	nvcc -std=c++11 -c -o caffe_c_g_r_u_step.cu.o caffe_c_g_r_u_step.cu.cc -I $(TF_INC) -I /usr/local/cuda-8.0/include/ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
	g++ -std=c++11 -shared -o caffe_c_g_r_u3.so caffe_c_g_r_u_step.cc caffe_c_g_r_u_step.cu.o -I /usr/local/cuda-8.0/include/ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart -lcudnn -lcurand -L /usr/local/cuda-8.0/lib64/

caffecgru2: caffe_c_g_r_u_step.cc
	$(eval TF_INC := $(shell python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'))
	nvcc -std=c++11 -c -o caffe_c_g_r_u_step.cu.o caffe_c_g_r_u_step.cu.cc -I $(TF_INC) -I /usr/local/cuda-8.0/include/ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
	g++ -std=c++11 -shared -o caffe_c_g_r_u2.so caffe_c_g_r_u_step.cc caffe_c_g_r_u_step.cu.o -I /usr/local/cuda-8.0/include/ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart -lcudnn -lcurand -L /usr/local/cuda-8.0/lib64/

caffecgrugradient3: caffe_c_g_r_u_gradient_step.cc
	$(eval TF_INC := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'))
	nvcc -std=c++11 -c -o caffe_c_g_r_u_step.cu.o caffe_c_g_r_u_step.cu.cc -I $(TF_INC) -I /usr/local/cuda-8.0/include/ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
	g++ -std=c++11 -shared -o caffe_c_g_r_u_gradient3.so caffe_c_g_r_u_gradient_step.cc caffe_c_g_r_u_step.cu.o -I /usr/local/cuda-8.0/include/ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart -lcudnn -lcurand -L /usr/local/cuda-8.0/lib64/
	
caffecgrugradient2: caffe_c_g_r_u_gradient_step.cc
	$(eval TF_INC := $(shell python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'))
	nvcc -std=c++11 -c -o caffe_c_g_r_u_step.cu.o caffe_c_g_r_u_step.cu.cc -I $(TF_INC) -I /usr/local/cuda-8.0/include/ -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
	g++ -std=c++11 -shared -o caffe_c_g_r_u_gradient2.so caffe_c_g_r_u_gradient_step.cc caffe_c_g_r_u_step.cu.o -I /usr/local/cuda-8.0/include/ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -lcudart -lcudnn -lcurand -L /usr/local/cuda-8.0/lib64/
