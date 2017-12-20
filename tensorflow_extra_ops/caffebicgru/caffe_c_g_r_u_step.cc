#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "caffe_c_g_r_u_step.h"
using namespace tensorflow;

REGISTER_OP("CaffeCGRUStepOp")
	.Input("input: float")
	.Input("z: Ref(float)")
	.Input("r: Ref(float)")
        .Input("deltar: Ref(float)")
	.Input("ht: Ref(float)")
	.Input("filterx: float")
	.Input("filterh: float")
	.Input("bias: float")
	.Input("hd: Ref(float)")
//	.Input("deltahd: Ref(float)")
	.Output("output: float")
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
           ::tensorflow::shape_inference::ShapeHandle out_shape;
           ::tensorflow::shape_inference::ShapeHandle temp_shape;
           ::tensorflow::shape_inference::ShapeHandle temp_shape2;
           c->Subshape(c->input(1),1,3,&temp_shape);
           c->Subshape(c->input(0),2,&temp_shape2);
           c->Concatenate(temp_shape,temp_shape2,&out_shape);
           c->set_output(0, out_shape);
           return Status::OK();
        });/*
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });*/


/*class CaffeCGRUStepOp : public CaffeCGRUCommonOp {
 public:
  //explicit CaffeCGRUStepOp(OpKernelConstruction* context) : OpKernel(context) {}
  explicit CaffeCGRUStepOp(OpKernelConstruction* context) : CaffeCGRUCommonOp(context){}
  void Compute(OpKernelContext* context) override {
    printf("not implemented\n");
  }
};*/

class CaffeCGRUGPUStepOp : public CaffeCGRUCommonOp {
public:
    CuDNNCGRU<float> *mycgru_;
  explicit CaffeCGRUGPUStepOp(OpKernelConstruction* context) : CaffeCGRUCommonOp(context){
	  this->mycgru_ = new CuDNNCGRU<float>(this->N_, this->inC_, this->outC_, this->X_, this->Y_, this->Z_, this->dimension_, this->Fsy_, this->Fsz_, 0, 0, 0, this->favorspeedovermemory_, false, false);
  }

  ~CaffeCGRUGPUStepOp() {
	  delete this->mycgru_;
  }

  void Compute(OpKernelContext* context) override {
    //printf("start compute\n");
    // Grab the input tensor

    auto input = context->input(0).flat<float>();
    //printf("start compute 2\n");

    mutex_lock lockz(*context->input_ref_mutex(1));    
    //printf("start compute 2.5\n");
    mutex_lock lockr(*context->input_ref_mutex(2));
    mutex_lock lockdeltard(*context->input_ref_mutex(3));
    mutex_lock lockht(*context->input_ref_mutex(4));

    auto z = context->mutable_input(1,true).flat<float>();
    //printf("start compute 3\n");
    auto r = context->mutable_input(2,true).flat<float>();
    auto deltard = context->mutable_input(3,true).flat<float>();
    auto ht = context->mutable_input(4,true).flat<float>();

    auto filterx = context->input(5).flat<float>();
    auto filterh = context->input(6).flat<float>();
    auto bias = context->input(7).flat<float>();    

    mutex_lock lockhd(*context->input_ref_mutex(8));
    auto hd = context->mutable_input(8,true).flat<float>();
    //mutex_lock lockdeltahd(*context->input_ref_mutex(9));
    //auto deltahd = context->input(9).flat<float>();
	// Create an output tensor
	Tensor* output_tensor = NULL;
        TensorShape output_shape({this->N_,this->outC_,this->X_,this->Y_,this->Z_});
	OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                &output_tensor));
	//printf("forward\n");
    this->mycgru_->Forward(input.data(),output_tensor->flat<float>().data(),hd.data(),z.data(),r.data(),deltard.data(),ht.data(),filterx.data(),filterh.data(),bias.data());
	//printf("done forward\n");  
}
};
//REGISTER_KERNEL_BUILDER(Name("CaffeCGRUStepOp").Device(DEVICE_CPU), CaffeCGRUStepOp);

REGISTER_KERNEL_BUILDER(Name("CaffeCGRUStepOp").Device(DEVICE_GPU), CaffeCGRUGPUStepOp);





