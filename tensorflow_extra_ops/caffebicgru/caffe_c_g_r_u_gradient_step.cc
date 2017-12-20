#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "caffe_c_g_r_u_step.h"
using namespace tensorflow;

REGISTER_OP("CaffeCGRUGradientStepOp")
	.Input("x: float")
	.Input("z: Ref(float)")
	.Input("r: Ref(float)")
        .Input("deltar: Ref(float)")
	.Input("ht: Ref(float)")
	.Input("filterx: float")
	.Input("filterh: float")
	.Input("bias: float")
        .Input("hd: Ref(float)")
        .Input("deltah: float")
//        .Input("deltahd: Ref(float)")
	.Output("deltax: float")
	.Output("deltafx: float")
	.Output("deltafh: float")
	.Output("deltab: float")
	.Attr("dimension: int = 0")
	.Attr("outC: int")
	.Attr("inC: int")
	.Attr("X: int")
	.Attr("Y: int")
	.Attr("Z: int")
	.Attr("fsy: int")
	.Attr("fsz: int")
	.Attr("favorspeedovermemory: bool")
	.Attr("N: int = 1")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(5));
      c->set_output(2, c->input(6));
      c->set_output(3, c->input(7));
      return Status::OK();
    });

/*
class CaffeCGRUStepOp : public CaffeCGRUCommonOp {
 public:
  //explicit CaffeCGRUStepOp(OpKernelConstruction* context) : OpKernel(context) {}
  explicit CaffeCGRUStepOp(OpKernelConstruction* context) : CaffeCGRUCommonOp(context){}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
	const Tensor& input_tensor = context->input(0);
	auto input = input_tensor.flat<float>();

	auto z = context->input(1).flat<float>();
	auto r = context->input(2).flat<float>();
        auto deltard = context->mutable_input(3,false).flat<float>();
	auto ht = context->input(3).flat<float>();

	auto filterx = context->input(4).flat<float>();
	auto filterh = context->input(5).flat<float>();
	auto bias = context->input(6).flat<float>();

	// Create an output tensor
        Tensor* output_tensor = NULL;
        TensorShape output_shape({2,this->N_,this->outC_,this->X_,this->Y_,this->Z_});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                &output_tensor));

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
	auto output = output_tensor->flat<float>();
    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};*/

class CaffeCGRUGPUGradientStepOp : public CaffeCGRUCommonOp {
public:
    CuDNNCGRU<float> *mycgru_;
  explicit CaffeCGRUGPUGradientStepOp(OpKernelConstruction* context) : CaffeCGRUCommonOp(context){
	  this->mycgru_ = new CuDNNCGRU<float>(this->N_, this->inC_, this->outC_, this->X_, this->Y_, this->Z_, this->dimension_, this->Fsy_, this->Fsz_, 0, 0, 0,this->favorspeedovermemory_, false,false);
  }

  ~CaffeCGRUGPUGradientStepOp() {
	  delete this->mycgru_;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor

    auto x = context->input(0).flat<float>();

    mutex_lock lockz(*context->input_ref_mutex(1));    
    auto z = context->mutable_input(1,true).flat<float>();

    mutex_lock lockr(*context->input_ref_mutex(2));    
    auto r = context->mutable_input(2,true).flat<float>();

    mutex_lock lockdeltard(*context->input_ref_mutex(3));    
    auto deltard = context->mutable_input(3,true).flat<float>();

    mutex_lock lockht(*context->input_ref_mutex(4));    
    auto ht = context->mutable_input(4,true).flat<float>();

    auto filterx = context->input(5).flat<float>();
    auto filterh = context->input(6).flat<float>();
    auto bias = context->input(7).flat<float>();

    mutex_lock lockhd(*context->input_ref_mutex(8));    
    auto hd = context->mutable_input(8, true).flat<float>();

    auto deltah = context->input(9).flat<float>();

    //mutex_lock lockdeltahd(*context->input_ref_mutex(10));    
    //auto deltahd = context->mutable_input(10, true).flat<float>();
    
	// Create an output tensor
	Tensor* deltax_tensor = NULL;
        TensorShape deltax_shape({this->N_,this->inC_,this->X_,this->Y_,this->Z_});
	OP_REQUIRES_OK(context, context->allocate_output(0, deltax_shape,
                                                &deltax_tensor));
Tensor* deltafx_tensor = NULL;
        TensorShape deltafx_shape({2,3,this->outC_,this->inC_,this->Fsy_,this->Fsz_});
	OP_REQUIRES_OK(context, context->allocate_output(1, deltafx_shape,
                                                &deltafx_tensor));
Tensor* deltafh_tensor = NULL;
        TensorShape deltafh_shape({2,3,this->outC_,this->outC_,this->Fsy_,this->Fsz_});
	OP_REQUIRES_OK(context, context->allocate_output(2, deltafh_shape,
                                                &deltafh_tensor));
Tensor* deltab_tensor = NULL;
        TensorShape deltab_shape({2,3,this->outC_});
	OP_REQUIRES_OK(context, context->allocate_output(3, deltab_shape,
                                                &deltab_tensor));

    //temporary data:
    Tensor deltahd_tensor;
    if (this->favorspeedovermemory_) {
        OP_REQUIRES_OK(
        context, context->allocate_temp(
                 DT_FLOAT,
                 TensorShape({2,this->N_,this->outC_,this->mycgru_->Outer_,this->mycgru_->Inner1_,this->mycgru_->Inner2_}),
                 &deltahd_tensor));
    } else {
       OP_REQUIRES_OK(
        context, context->allocate_temp(
                 DT_FLOAT,
                 TensorShape({2,this->N_,this->outC_,this->mycgru_->Inner1_,this->mycgru_->Inner2_}),
                 &deltahd_tensor));
    }
    auto deltahd = deltahd_tensor.flat<float>();
//(Dtype * deltah, Dtype * x,Dtype * hd, Dtype * zd, Dtype * rd, Dtype * deltard, Dtype * htilded, Dtype * deltahd, Dtype * fx, Dtype * fh, Dtype * b, Dtype * deltafx, Dtype * deltafh, Dtype * deltab)
	//printf("backward\n");
    this->mycgru_->Backward(deltah.data(),x.data(),hd.data(),z.data(),r.data(),deltard.data(),ht.data(),deltahd.data(),filterx.data(),filterh.data(),bias.data(),deltafx_tensor->flat<float>().data(),deltafh_tensor->flat<float>().data(),deltab_tensor->flat<float>().data(),deltax_tensor->flat<float>().data());
	//printf("done backward\n");
  }
};
//REGISTER_KERNEL_BUILDER(Name("CaffeCGRUStepOp").Device(DEVICE_CPU), CaffeCGRUStepOp);

REGISTER_KERNEL_BUILDER(Name("CaffeCGRUGradientStepOp").Device(DEVICE_GPU), CaffeCGRUGPUGradientStepOp);



