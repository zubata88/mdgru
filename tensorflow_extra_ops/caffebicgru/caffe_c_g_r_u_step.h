#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cudnn.h>
#include <curand.h>

namespace tensorflow {
template <typename Dtype>
class CRNN { //BIDIRECTIONAL CRNN ABSTRACT CLASS
protected:
public:
    int N_;
    int inC_;
    int outC_;
    int X_;
    int Y_;
    int Z_;
    int direction_;
    int Fsy_;
    int Fsz_;
    int Py_;
    int Pz_;
    float dropconnectx_;
    bool favorspeedovermemory_;
    bool bnx_;
    bool bnh_;
    int Inner1_;
    int Inner2_;
    int Outer_;
    int Inner1Stride_;
    int Inner2Stride_;
    int OuterStride_;
//    shared_ptr<Blob<Dtype> > hd_;
//    shared_ptr<Blob<Dtype> > xd_; //only used in conjunction with bnx
//    bool normalizeByDepth_;
//    shared_ptr<Blob<Dtype> > bn_variance_x_; //{d,outer,inc}
//    shared_ptr<Blob<Dtype> > bn_mean_x_; //{d,outer,inc}
//    shared_ptr<Blob<Dtype> > bn_moving_meanvar_x_; //{d,meanorvariance,inc}
//    shared_ptr<Blob<Dtype> > bn_moving_meanvar_number_x_; //{1}
//    Dtype bn_epsilon_x_;
//    shared_ptr<Blob<Dtype> > bn_variance_h_;//{d,outer,outc}
//    shared_ptr<Blob<Dtype> > bn_mean_h_; //{d,outer,outc}
//    shared_ptr<Blob<Dtype> > bn_moving_meanvar_h_;//{d,outc}
//    shared_ptr<Blob<Dtype> > bn_moving_meanvar_number_h_;//{1}
//    Dtype bn_epsilon_h_;
//    Dtype bn_moving_average_factor_;
//    bool bn_global_stats_;
    CRNN(int N, int inC, int outC, int X, int Y, int Z, int direction, int filter_size_a, int filter_size_b, int padding_a, int padding_b, float dropconnectx, int favorspeedovermemory, bool bnx, bool bnh)
        :N_(N), inC_(inC),outC_(outC), X_(X),Y_(Y),Z_(Z),direction_(direction), Fsy_(filter_size_a),Fsz_(filter_size_b),Py_(padding_a),Pz_(padding_b),dropconnectx_(dropconnectx),favorspeedovermemory_(favorspeedovermemory), bnx_(bnx), bnh_(bnh){
//        normalizeByDepth_ = false;
        //vector<int> shapeout = {2,N_,outC_,X_,Y_,Z_};
        //hd_.reset(new Blob<Dtype>(shapeout));
        switch(direction){
        case 0:
            Outer_ = X_;
            Inner1_ = Y_;
            Inner2_ = Z_;
            OuterStride_ = Y_*Z_;
            Inner1Stride_ = Z_;
            Inner2Stride_ = 1;
            break;
        case 1:
            Outer_ = Y_;
            Inner1_ = X_;
            Inner2_ = Z_;
            OuterStride_ = Z_;
            Inner1Stride_ = Y_*Z_;
            Inner2Stride_ = 1;
            break;
        case 2:
            Outer_ = Z_;
            Inner1_ = X_;
            Inner2_ = Y_;
            OuterStride_ = 1;
            Inner1Stride_ = Y_*Z_;
            Inner2Stride_ = Z_;
            break;
        }
//        if (this->bnx_) {
//            vector<int> vecshape = {2,N_,inC_,Inner1_,Inner2_};
//            this->xd_.reset(new Blob<Dtype>(vecshape));
//            //FIXME: set up remaining x blobs (from outside?)
//        }
    }
//    virtual void Forward(const vector<Blob<Dtype> *>& bottom, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& top) {printf("abstract\n"); throw 131;}
//    virtual void Backward(const vector<Blob<Dtype> *>& top, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& bottom, bool propagate_down){printf("abstract\n"); throw 131;}
//    virtual void Forward_cpu(const vector<Blob<Dtype> *>& bottom, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& top){printf("abstract\n"); throw 131;}
//    virtual void Backward_cpu(const vector<Blob<Dtype> *>& top, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& bottom, bool propagate_down){printf("abstract\n"); throw 131;}
};

template <typename Dtype>
class CGRU : public CRNN<Dtype>{
public:
//    shared_ptr<Blob<Dtype> > zd_;
//    shared_ptr<Blob<Dtype> > rd_;
//    shared_ptr<Blob<Dtype> > htilded_;
//    shared_ptr<Blob<float> > dropconnectxmask_;
//    shared_ptr<Blob<Dtype> > dropconnectxweights_;
    CGRU(int N, int inC, int outC, int X, int Y, int Z, int direction, int filter_size_y, int filter_size_z, int padding_y, int padding_z, float dropconnectx, int favorspeedovermemory, bool bnx, bool bnh)
        :CRNN<Dtype>(N,inC,outC,X,Y,Z,direction,filter_size_y,filter_size_z,padding_y,padding_z,dropconnectx,favorspeedovermemory, bnx, bnh){
//        vector<int> shapetemporarys = this->hd_->shape();
//        CHECK_GT(shapetemporarys.size(), 3+this->direction_) << "corresponding dimension to direction not available\n";
//        if (favorspeedovermemory == false) {
//            shapetemporarys[3+this->direction_] = 1; //we dont need more than 1 along the direction we're going. saves memory!!!
//        }
//        if (this->dropconnectx_ > 0) {
//            vector<int> weightxshape = {2,3,outC,inC,filter_size_y,filter_size_z};
//            this->dropconnectxmask_.reset(new Blob<float>(weightxshape));
//            this->dropconnectxweights_.reset(new Blob<Dtype>(weightxshape));
//        }

//        zd_.reset(new Blob<Dtype>(shapetemporarys));
//        rd_.reset(new Blob<Dtype>(shapetemporarys));
//        htilded_.reset(new Blob<Dtype>(shapetemporarys));
//        if (this->bnx_) {
//            shapetemporarys[3+this->direction_] = 1; //we dont need more than 1 along the direction we're going. saves memory!!!
//            shapetemporarys[1] = this->inC_; //this is for xd so we need the input channel number!
//            this->xd_.reset(new Blob<Dtype>(shapetemporarys));
//        }
    }
    //void doForwardConvolutionsOneDirectionForZdRdHtilded_cpu(Dtype * zd,Dtype * rd,Dtype * htilded,Dtype * deltard,const Dtype * x,const Dtype * hd,const Dtype * fx,const Dtype * fh,const Dtype * b,int preouter,int outer,int d);

    //virtual void Forward_cpu(const vector<Blob<Dtype> *>& bottom, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& top);
    //virtual void Backward_cpu(const vector<Blob<Dtype> *>& top, shared_ptr<Blob<Dtype> >& weightsx,shared_ptr<Blob<Dtype> >& weightsh,shared_ptr<Blob<Dtype> >& bias,const vector<Blob<Dtype>* >& bottom, bool propagate_down);
};

template <typename Dtype>
class CuDNNCGRU : public CGRU<Dtype>{
public:
        CuDNNCGRU(int N, int inC, int outC, int X, int Y, int Z, int direction, int filter_size_y,
                    int filter_size_z, int padding_y, int padding_z, float dropconnectx, int favorspeedovermemory, bool bnx, bool bnh);
    void Forward(const Dtype * x, Dtype * h,Dtype * hd, Dtype * zd, Dtype * rd, Dtype * deltard, Dtype * htilded, const Dtype * fx, const Dtype * fh, const Dtype * b);
    void Backward(const Dtype * deltah, const Dtype * x,Dtype * hd, Dtype * zd, Dtype * rd, Dtype * deltard, Dtype * htilded, Dtype * deltahd, const Dtype * fx, const Dtype * fh, const Dtype * b, Dtype * deltafx, Dtype * deltafh, Dtype * deltab, Dtype * deltax);
    void doForwardConvolutionsOneDirectionForZdRdHtilded_gpu(Dtype * zd,Dtype * rd,Dtype * htilded,Dtype * deltard, const Dtype * x,Dtype * hd, const Dtype * fx, const Dtype * fh, const Dtype * b,int outer,int preouter,int d);

    cudnnDataType_t getCudnnDtype() {
        switch(sizeof(Dtype)){
        case 4:         return CUDNN_DATA_FLOAT;
        case 8:         return CUDNN_DATA_DOUBLE;
        case 2:         return CUDNN_DATA_HALF;
        default:        return CUDNN_DATA_FLOAT;
        }
    }
    cudnnTensorDescriptor_t bottomXTensor_,bottomHTensor_, topTempTensor_,topBiasTensor_,topHTensor_, biasTensor_;
    cudnnFilterDescriptor_t filterXDesc_, filterHDesc_;
    cudnnConvolutionDescriptor_t convXDesc_,convHDesc_, convTempDesc_;
    cudaStream_t* stream_;
    cudnnHandle_t* cudnnHandle_;
    cudnnConvolutionFwdAlgo_t fwdXalgo_,fwdHalgo_;
    cudnnConvolutionBwdDataAlgo_t backwdHd_;
    cudnnConvolutionBwdDataAlgo_t backwdXd_;
    cudnnConvolutionBwdFilterAlgo_t backwdHf_;
    cudnnConvolutionBwdFilterAlgo_t backwdXf_;
    curandGenerator_t generator_;
    size_t workspaceSize_;
    size_t workspaceSizeN_;
    void ** workspace_;
    void * workspaceData_;
    void ** workspaceN_;
    void * workspaceDataN_;

};


class CaffeCGRUCommonOp : public OpKernel {
protected:
    int N_;
    int inC_;
    int outC_;
    int X_;
    int Y_;
    int Z_;
    int direction_;
    int Fsy_;
    int Fsz_;
    int Py_;
    int Pz_;
    float dropconnectx_;
    bool favorspeedovermemory_;
    bool bnx_;
    bool bnh_;
    int Inner1_;
    int Inner2_;
    int Outer_;
    int Inner1Stride_;
    int Inner2Stride_;
    int OuterStride_;
    int dimension_;

 public:
  explicit CaffeCGRUCommonOp(OpKernelConstruction* context) : OpKernel(context) {
	    // Get the index of the value to preserve
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("dimension", &dimension_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("outC", &outC_));
printf("outC: %d",outC_);
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("inC", &inC_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("X", &X_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("Y", &Y_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("Z", &Z_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("fsy", &Fsy_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("fsz", &Fsz_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("N", &N_));
	    OP_REQUIRES_OK(context,
	                   context->GetAttr("favorspeedovermemory", &favorspeedovermemory_));
	    // Check that preserve_index is positive
	    OP_REQUIRES(context, dimension_ >= 0,
	                errors::InvalidArgument("Need dimension >= 0, got ",
	                			dimension_));

  }
};
};
