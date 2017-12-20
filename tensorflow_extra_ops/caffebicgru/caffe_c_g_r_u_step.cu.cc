
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>
#include <curand.h>
#include "caffe_c_g_r_u_step.h"

using namespace tensorflow;
#define cudnnErrchk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
#define checkCUDNN(ans) {cudnnErrchk(ans)}
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define EXCEPTION 20;

#define CUDNN_CHECK(ans) {checkCUDNN(ans);}


#define MY_CHECK_EQ(val1, val2) {test_for_eq(val1,val2);}

inline void test_for_eq(int val1,int val2) {
    if (val1 != val2) {
        printf("val1 and val2 should be equal!\n");
        throw 131;
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
          exit(code);
      }
   }
}
template <typename Dtype>
__device__ __forceinline__ Dtype sigma(Dtype & x) {
    return 1.0/(1 + exp(-x));
}

template <typename Dtype>
__device__ __forceinline__ void cgrus_compute_gates_forward(Dtype & zd, Dtype & rd, Dtype & deltard, Dtype & htilded, int preouter) {
    zd = sigma(zd);
    rd = sigma(rd);
    if (preouter > 0) {
        htilded += rd*deltard;
    }
    htilded = tanh(htilded);
}

template <typename Dtype>
__global__ void cgrus_forwardnoconv(Dtype* hd, Dtype * h, Dtype* zd, Dtype* rd, Dtype * deltard, Dtype* htilded,
                                     int outer, int preouter,int N, int outC, int Outer, int Inner1, int Inner2, int Inner1Stride, int Inner2Stride, int OuterStride, int d){
    int n = blockIdx.x;
    int c = blockIdx.y;
    int inner1 = blockIdx.z;
    int inner2 = threadIdx.x;

    int temploc = ((n*outC+c)*Inner1+inner1)*Inner2+inner2;
    int hloc = (n*outC+c)*Outer*Inner1*Inner2+outer*OuterStride+inner1*Inner1Stride+inner2*Inner2Stride;
    int hdloc = ((((d*N+n)*outC+c)*Outer+outer)*Inner1+inner1)*Inner2+inner2;//*Inner1*Inner2+outer*OuterStride+inner1*Inner1Stride+inner2*Inner2Stride;

    cgrus_compute_gates_forward(zd[temploc], rd[temploc], deltard[temploc], htilded[temploc], preouter);

    hd[hdloc] = (1-zd[temploc])*htilded[temploc];
    if (preouter > 0) {
        hd[hdloc] += zd[temploc]*hd[hdloc-(1-2*d)*Inner1*Inner2]; //if d == 1, we do +OuterStride
    }
    h[hloc] += hd[hdloc];

}



namespace cudnn {
template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);

}  // namespace cudnn

template <typename Dtype>
CuDNNCGRU<Dtype>::CuDNNCGRU(int N, int inC, int outC, int X, int Y, int Z, int direction, int filter_size_y,
                    int filter_size_z, int padding_y, int padding_z, float dropconnectx, int favorspeedovermemory, bool bnx, bool bnh)
            :CGRU<Dtype>(N,inC,outC,X,Y,Z,direction, filter_size_y,filter_size_z,padding_y,padding_z,dropconnectx,favorspeedovermemory, bnx, bnh){

            int G = 2*3; //2 directions, 3 convolutions each?

            stream_ = new cudaStream_t[G];
            cudnnHandle_ = new cudnnHandle_t[G];
            workspaceN_ = new void*[G];
            workspace_ = new void*[G];

            for (int g = 0; g < G; g++) {
                gpuErrchk(cudaStreamCreate(&stream_[g]));
                checkCUDNN(cudnnCreate(&cudnnHandle_[g]));
                checkCUDNN(cudnnSetStream(cudnnHandle_[g], stream_[g]));
                workspace_[g] = NULL;
                workspaceN_[g] = NULL;
            }
            /*if (this->dropconnectx_ > 0) {
                curandCreateGenerator(&this->generator_, CURAND_RNG_PSEUDO_DEFAULT);
                curandSetPseudoRandomGeneratorSeed(this->generator_, 10101010);
            }*/

            checkCUDNN(cudnnCreateTensorDescriptor(&bottomXTensor_));
            checkCUDNN(cudnnCreateTensorDescriptor(&bottomHTensor_));
            checkCUDNN(cudnnCreateTensorDescriptor(&topTempTensor_));
            //checkCUDNN(cudnnCreateTensorDescriptor(&topBiasTensor_));
            //checkCUDNN(cudnnCreateTensorDescriptor(&topHTensor_));
            checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor_));

            checkCUDNN(cudnnCreateFilterDescriptor(&filterXDesc_));
            checkCUDNN(cudnnCreateFilterDescriptor(&filterHDesc_));

            checkCUDNN(cudnnCreateConvolutionDescriptor(&convXDesc_));
            checkCUDNN(cudnnCreateConvolutionDescriptor(&convHDesc_));

            checkCUDNN(cudnnSetTensor4dDescriptor(biasTensor_,
                                                  CUDNN_TENSOR_NCHW,
                                                  getCudnnDtype(),
                                                  1, this->outC_,
                                                  1, 1));

            if (this->bnx_) {
                printf("bottomXtensor like temphtop but inC instead of outC\n");
                checkCUDNN(cudnnSetTensor4dDescriptor(bottomXTensor_,
                                                    CUDNN_TENSOR_NCHW,
                                                    getCudnnDtype(),
                                                    N, this->inC_,
                                                    this->Inner1_, this->Inner2_));
            } else {
                checkCUDNN(cudnnSetTensor4dDescriptorEx(bottomXTensor_,
                                                          getCudnnDtype(),
                                                          N,inC,
                                                          this->Inner1_, this->Inner2_,inC*X*Y*Z,X*Y*Z, this->Inner1Stride_, this->Inner2Stride_));

            }

            checkCUDNN(cudnnSetTensor4dDescriptorEx(bottomHTensor_,
                                                  getCudnnDtype(),
                                                  N,outC,
                                                  this->Inner1_, this->Inner2_, outC*X*Y*Z, X*Y*Z, this->Inner2_, 1));
            //we reorder hd to be in the shape DxNxoutCxOuterxInner1xInner2

            checkCUDNN(cudnnSetFilter4dDescriptor(filterXDesc_,
                                                  getCudnnDtype(),
                                                  CUDNN_TENSOR_NCHW,
                                                  outC,
                                                  inC,
                                                  filter_size_y,
                                                  filter_size_z));
            checkCUDNN(cudnnSetFilter4dDescriptor(filterHDesc_,
                                                  getCudnnDtype(),
                                                  CUDNN_TENSOR_NCHW,
                                                  outC,
                                                  outC,
                                                  filter_size_y,
                                                  filter_size_z));

            checkCUDNN(cudnnSetConvolution2dDescriptor(convXDesc_,
                                                       filter_size_y/2, filter_size_z/2,
                                                       1, 1,
                                                       1, 1,
                                                       CUDNN_CROSS_CORRELATION));
            checkCUDNN(cudnnSetConvolution2dDescriptor(convHDesc_,
                                                       filter_size_y/2, filter_size_z/2,
                                                       1, 1,
                                                       1, 1,
                                                       CUDNN_CROSS_CORRELATION));
            //printf("old n%d c%d h%d w%d\n",N,inC,this->Inner1_,this->Inner2_);
            int nN,nC,nH,nW;
            checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convXDesc_,
                                                             bottomXTensor_,
                                                             filterXDesc_,
                                                             &nN, &nC, &nH, &nW));
            //printf("new for x n%d c%d h%d w%d\n",nN,nC,nH,nW);
            MY_CHECK_EQ(nN,N);
            MY_CHECK_EQ(nC,outC);
            MY_CHECK_EQ(nH,this->Inner1_);
            MY_CHECK_EQ(nW,this->Inner2_);

                checkCUDNN(cudnnSetTensor4dDescriptor(topTempTensor_,
                                                    CUDNN_TENSOR_NCHW,
                                                    getCudnnDtype(),
                                                    N, this->outC_,
                                                    this->Inner1_, this->Inner2_));

        //        checkCUDNN(cudnnSetTensor4dDescriptor(topBiasTensor_,
        //                                                CUDNN_TENSOR_NCHW,
        //                                              getCudnnDtype(),
        //                                                N, this->outC_,
        //                                                Y, Z));

            checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convHDesc_,
                                                             bottomHTensor_,
                                                             filterHDesc_,
                                                             &nN, &nC, &nH, &nW));
            //printf("new for h n%d c%d h%d w%d\n",nN,nC,nH,nW);
            MY_CHECK_EQ(nN,N);
            MY_CHECK_EQ(nC,outC);
            MY_CHECK_EQ(nH,this->Inner1_);
            MY_CHECK_EQ(nW,this->Inner2_);
        //        checkCUDNN(cudnnSetTensor4dDescriptor(topHTensor_,
        //                                              CUDNN_TENSOR_NCHW,
        //                                                getCudnnDtype(),
        //                                                nN,nC,nH,nW));
        //        checkCUDNN(cudnnSetTensor4dDescriptor(topHTensor_,
        //                                                CUDNN_TENSOR_NCHW,
        //                                                getCudnnDtype(),
        //                                                N, this->outC_,
        //                                                Inner1, Inner2));

            size_t workspace_limit_bytes = 8*1024*1024;

        workspaceSizeN_ = 0;
        workspaceSize_ = 0;
            checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle_[0],
                                                           bottomXTensor_,
                                                           filterXDesc_,
                                                           convXDesc_,
                                                           topTempTensor_,
                                                           CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                                                           //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                           workspace_limit_bytes,
                                                           &fwdXalgo_));

            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_[0],
                                                               bottomXTensor_,
                                                               filterXDesc_,
                                                               convXDesc_,
                                                               topTempTensor_,
                                                               fwdXalgo_,
                                                               &workspaceSizeN_));
            checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle_[0],
                                                           bottomHTensor_,
                                                           filterHDesc_,
                                                           convHDesc_,
                                                           topTempTensor_,
                                                           //CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                                                           //CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                           CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,

                                                           workspace_limit_bytes,
                                                           &fwdHalgo_));

            checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_[0],
                                                               bottomHTensor_,
                                                               filterHDesc_,
                                                               convHDesc_,
                                                               topTempTensor_,
                                                               fwdHalgo_,
                                                               &workspaceSize_));


            size_t tmpsize = 0;

            checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle_[0], bottomXTensor_, topTempTensor_, convXDesc_, filterXDesc_,
                       CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes, &backwdXf_));
            //CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &backwdXf_));

            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle_[0], bottomXTensor_, topTempTensor_, convXDesc_, filterXDesc_,
                backwdXf_, &tmpsize));
            workspaceSizeN_ = std::max(workspaceSizeN_, tmpsize);


            checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle_[0], topTempTensor_, topTempTensor_, convHDesc_, filterHDesc_,
            //           CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &backwdHf_));
                       //CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &backwdHf_));
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes, &backwdHf_));

            checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle_[0], topTempTensor_, topTempTensor_, convHDesc_, filterHDesc_,
                backwdHf_, &tmpsize));


            workspaceSize_ = std::max(workspaceSize_, tmpsize);


            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle_[0], filterXDesc_, topTempTensor_, convXDesc_, bottomXTensor_,
                       CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes, &backwdXd_));
            //CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &backwdXd_));

            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle_[0], filterXDesc_, topTempTensor_, convXDesc_, bottomXTensor_,
                backwdXd_, &tmpsize));

            workspaceSizeN_ = std::max(workspaceSizeN_, tmpsize);

            checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle_[0], filterHDesc_, topTempTensor_, convHDesc_, topTempTensor_,
                       //       CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &backwdHd_));
                              CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace_limit_bytes, &backwdHd_));
            //CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &backwdHd_));

            checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle_[0], filterHDesc_, topTempTensor_, convHDesc_, topTempTensor_,
                backwdHd_, &tmpsize));

            workspaceSize_ = std::max(workspaceSize_, tmpsize);


            if (workspaceSize_ > 0 || workspaceSizeN_ > 0) {
                gpuErrchk(cudaMalloc(&workspaceData_,workspaceSize_*G));
                gpuErrchk(cudaMalloc(&workspaceDataN_,workspaceSizeN_*G));
                for (int g = 0; g < G; g++) {
                    workspace_[g] = reinterpret_cast<char *>(workspaceData_) + g*workspaceSize_;
                    workspaceN_[g] = reinterpret_cast<char *>(workspaceDataN_) + g*workspaceSizeN_;
                }
            }
            printf("wasting %d*(%d+%d)=%d bytes for workspace\n", G, workspaceSize_, workspaceSizeN_,G*(workspaceSize_+workspaceSizeN_) );

        }



template <typename Dtype>
void CuDNNCGRU<Dtype>::doForwardConvolutionsOneDirectionForZdRdHtilded_gpu(Dtype * zd,Dtype * rd,Dtype * htilded,Dtype * deltard,const Dtype * xAtOuter,Dtype * hd,const Dtype * fx, const Dtype * fh,const Dtype * b,int outer,int preouter,int d) {
    const int CoCiFsyFsz = this->outC_*this->inC_*this->Fsy_*this->Fsz_;
    const int CoCoFsyFsz = this->outC_*this->outC_*this->Fsy_*this->Fsz_;
    const int NCoYZForTemps = this->N_*this->outC_*this->Inner1_*this->Inner2_;
    const int NCoXYZ = this->N_*this->outC_*this->Outer_*this->Inner1_*this->Inner2_;//this->OuterStride_*NCoYZ;

    int tempoffset = 0;
    if (this->favorspeedovermemory_) {
        tempoffset = (outer*2+d)*NCoYZForTemps;
    } else {
        tempoffset = d*NCoYZForTemps;
    }
    CUDNN_CHECK(cudnnAddTensor(this->cudnnHandle_[d*3],cudnn::dataType<Dtype>::one,this->biasTensor_, b+(d*3+0)*this->outC_,cudnn::dataType<Dtype>::zero,
          this->topTempTensor_,zd+tempoffset));
    CUDNN_CHECK(cudnnAddTensor(this->cudnnHandle_[d*3+1],cudnn::dataType<Dtype>::one,this->biasTensor_, b+(d*3+1)*this->outC_,cudnn::dataType<Dtype>::zero,
          this->topTempTensor_,rd+tempoffset));
    CUDNN_CHECK(cudnnAddTensor(this->cudnnHandle_[d*3+2],cudnn::dataType<Dtype>::one,this->biasTensor_, b+(d*3+2)*this->outC_,cudnn::dataType<Dtype>::zero,
          this->topTempTensor_,htilded+tempoffset));
    //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
    //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+1]));
    //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+2]));


    CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3],cudnn::dataType<Dtype>::one, this->bottomXTensor_,xAtOuter,this->filterXDesc_,fx+(d*3)*CoCiFsyFsz,
                this->convXDesc_, this->fwdXalgo_, this->workspaceN_[d*3], this->workspaceSizeN_, cudnn::dataType<Dtype>::one, this->topTempTensor_, zd+tempoffset));
    CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3+1],cudnn::dataType<Dtype>::one, this->bottomXTensor_,xAtOuter,this->filterXDesc_,fx+(d*3+1)*CoCiFsyFsz,
                this->convXDesc_, this->fwdXalgo_, this->workspaceN_[d*3+1], this->workspaceSizeN_, cudnn::dataType<Dtype>::one, this->topTempTensor_, rd+tempoffset));
    CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3+2],cudnn::dataType<Dtype>::one, this->bottomXTensor_,xAtOuter,this->filterXDesc_,fx+(d*3+2)*CoCiFsyFsz,
                this->convXDesc_, this->fwdXalgo_, this->workspaceN_[d*3+2], this->workspaceSizeN_, cudnn::dataType<Dtype>::one, this->topTempTensor_, htilded+tempoffset));

    if (preouter > 0) {
        //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
        //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+1]));
        CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3],cudnn::dataType<Dtype>::one, this->bottomHTensor_,hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_,this->filterHDesc_,fh+(d*3)*CoCoFsyFsz,
                    this->convHDesc_, this->fwdHalgo_, this->workspace_[d*3], this->workspaceSize_, cudnn::dataType<Dtype>::one, this->topTempTensor_, zd+tempoffset));
        CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3+1],cudnn::dataType<Dtype>::one, this->bottomHTensor_,hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_,this->filterHDesc_,fh+(d*3+1)*CoCoFsyFsz,
                    this->convHDesc_, this->fwdHalgo_, this->workspace_[d*3+1], this->workspaceSize_, cudnn::dataType<Dtype>::one, this->topTempTensor_, rd+tempoffset));
        CUDNN_CHECK(cudnnConvolutionForward(this->cudnnHandle_[d*3+2],cudnn::dataType<Dtype>::one, this->bottomHTensor_,hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_,this->filterHDesc_,fh+(d*3+2)*CoCoFsyFsz,
                    this->convHDesc_, this->fwdHalgo_, this->workspace_[d*3+2], this->workspaceSize_, cudnn::dataType<Dtype>::zero, this->topTempTensor_, deltard+tempoffset));
        //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+2]));

    } else {
        cudaMemsetAsync(deltard+tempoffset, 0, NCoYZForTemps*sizeof(Dtype),this->stream_[d*3+2]);
    }
    gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
    gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+1]));
    gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+2]));

}

template <typename Dtype>
void CuDNNCGRU<Dtype>::Forward(const Dtype * x, Dtype * h, Dtype * hd, Dtype * zd, Dtype * rd, Dtype * deltard, Dtype * htilded, const Dtype * fx, const Dtype * fh, const Dtype * b){
    //Dtype * xd;
    //Dtype *xdatOuter;
    if (this->bnx_) {
        //xd = this->xd_->mutable_gpu_data();
    }
    //we dont need to memset rd,zd,deltard and htilded to 0, since cudnn routines overwrite the last part first.
    cudaMemset(h,0,this->N_*this->outC_*this->X_*this->Y_*this->Z_*sizeof(Dtype));
    cudaMemset(hd,0,2*this->N_*this->outC_*this->X_*this->Y_*this->Z_*sizeof(Dtype));

    //Dtype manx, manh;
    //if (this->bnx_) {
        //manx = this->bn_moving_meanvar_number_x_->mutable_cpu_data()[0];
    //}

    for (int preouter = 0; preouter < this->Outer_; preouter++) {

        for (int d = 0; d < 2; d++) {


            int outer = preouter;
            if (d==1) {
                outer = this->Outer_-1-preouter;
            }
            //if (this->bnx_) {
            //    xdatOuter = xd+d*this->Inner1_*this->Inner2_*this->inC_*this->N_;
            //} else {
                const Dtype * xdatOuter = x+outer*this->OuterStride_;
            //}

//            if (this->bnx_) {
//                dim3 blocks(this->inC_);
//                dim3 threads(128, 1, 1);
//                Dtype * meanvarx = this->bn_moving_meanvar_x_->mutable_gpu_data()+d*2*this->inC_;
//                Dtype * varx = this->bn_variance_x_->mutable_gpu_data()+(d*this->Outer_+outer)*this->inC_;
//                if (this->bn_global_stats_) {
//                    normalize_by_mean_and_variance_x_slice<<<blocks, threads>>>(x, xdatOuter, varx, meanvarx, meanvarx+this->inC_,
//                                                                                manx, this->bn_epsilon_x_, outer, this->Outer_, this->Inner1_, this->Inner2_,
//                                                                                this->OuterStride_, this->Inner1Stride_, this->Inner2Stride_, this->N_, this->inC_);
//                } else {
//                    Dtype * meanx = this->bn_mean_x_->mutable_gpu_data()+(d*this->Outer_+outer)*this->inC_;
//                    mergesum_128_compute_and_apply_expectation_value_x_slice<<<blocks, threads>>>(x,xdatOuter,meanx, meanvarx, this->bn_moving_average_factor_, outer,
//                                        this->Outer_, this->Inner1_, this->Inner2_, this->OuterStride_, this->Inner1Stride_, this->Inner2Stride_, this->N_, this->inC_);
//                    gpuErrchk(cudaDeviceSynchronize());
//                    gpuErrchk( cudaPeekAtLastError() );
//                    mergesum_128_compute_and_apply_variance<<<blocks, threads>>>(xdatOuter, varx, meanvarx+this->inC_, this->bn_moving_average_factor_,
//                                              this->bn_epsilon_x_, this->Inner1_*this->Inner2_, this->Inner1_*this->Inner2_, this->N_, this->inC_);
//                }
//                gpuErrchk(cudaDeviceSynchronize());
//                gpuErrchk( cudaPeekAtLastError() );
//            }
            doForwardConvolutionsOneDirectionForZdRdHtilded_gpu(zd,rd,htilded,deltard,xdatOuter,hd,fx,fh,b, outer, preouter, d);

            gpuErrchk( cudaPeekAtLastError() );
            dim3 blocks(this->N_, this->outC_, this->Inner1_);
            dim3 threads(this->Inner2_, 1, 1);
            int offset=this->N_*this->outC_*this->Inner1_*this->Inner2_;
            if (this->favorspeedovermemory_) {
                offset *= (outer*2+d);
            } else {
                offset *= d;
            }
            gpuErrchk( cudaDeviceSynchronize() );
            cgrus_forwardnoconv<<<blocks,threads>>>(hd,h,zd+offset,rd+offset, deltard+offset,htilded+offset,outer,preouter,this->N_, this->outC_,
                                                   this->Outer_, this->Inner1_, this->Inner2_, this->Inner1Stride_, this->Inner2Stride_,this->OuterStride_, d);


        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
//        if (!this->bn_global_stats_) {
//            if (this->bnx_) {
//                manx *= this->bn_moving_average_factor_; //this is down here since we only count it once per direction.
//                manx += 1;
//            }

//        }
    }
//    if (this->bnx_) {
//        this->bn_moving_meanvar_number_x_->mutable_cpu_data()[0] = manx;
//    }

}
    template <typename Dtype>
    __device__ __forceinline__ void mdrnn_bicgrus_backwardprepare_same(Dtype& zd, Dtype& htilded,
                                                                       Dtype& deltahd, Dtype& prevdeltahd, const Dtype& deltah, Dtype& prevhd, Dtype& rd, Dtype& deltard, int preouter, bool bnh) {
        //now we fetch the real error from the top layer at position t (after this point, it contains the upper information + the convos + (1-zd)*deltahd:
        if (!bnh) {
            deltahd += deltah;
        }
        Dtype deltazd = -htilded*deltahd;
        if (preouter > 0) {
            deltazd += +prevhd*deltahd;
        }
        Dtype deltahtilded = (1-zd)*deltahd;


        //lets now move one step back with deltahd from t ot t-1:
        if (preouter > 0) {
            prevdeltahd = zd*deltahd; //this is used next round!
        }
        //we put all the backpropagated values of z, r, htilde into the respective variables, as a preparation for the convolutions that follow.
        zd = zd*(1-zd)*deltazd;
        htilded = (1-htilded*htilded)*deltahtilded;
        deltard *= htilded * (1-rd)*rd; //deltard is the combination of actual dr * (1-r)*r
        rd *= htilded;
    }
    template <typename Dtype>
    __global__ void mdrnn_bicgrus_backwardprepare(Dtype* zd, Dtype* htilded,
                                                  Dtype* deltahd, const Dtype* deltah, Dtype * hd, Dtype * rd, Dtype * deltard,
                                                 int outer, int preouter, int N, int outC, int Inner1, int Inner2, int Outer,
                                            int Inner1Stride, int Inner2Stride, int OuterStride, int d, bool bnh) {

        int n = blockIdx.x;
        int c = blockIdx.y;
        int inner1 = blockIdx.z;
        int inner2 = threadIdx.x;

        int temploc = (((d*N+n)*outC+c)*Inner1+inner1)*Inner2+inner2;
        int hloc = (n*outC+c)*Outer*Inner1*Inner2+outer*OuterStride+inner1*Inner1Stride+inner2*Inner2Stride;
        int hdloctm1 = ((((d*N+n)*outC+c)*Outer+(outer-1+2*d))*Inner1+inner1)*Inner2+inner2;

        cgrus_compute_gates_forward(zd[temploc], rd[temploc], deltard[temploc], htilded[temploc], preouter);
        //mdrnn_bicgrus_backwardprepare_same(zd,htilded,deltahd,deltah,hd,rd,deltard,temploc,temploc,hloc,hdloctm1,preouter);
        mdrnn_bicgrus_backwardprepare_same(zd[temploc], htilded[temploc], deltahd[temploc], deltahd[temploc], deltah[hloc], hd[hdloctm1], rd[temploc], deltard[temploc], preouter, bnh);
    }
    template <typename Dtype>
    __global__ void mdrnn_bicgrus_backwardprepare_fast(Dtype* zd, Dtype* htilded,
                                                  Dtype* deltahd, const Dtype* deltah, Dtype * hd, Dtype * rd, Dtype * deltard,
                                                 int outer, int preouter, int N, int outC, int Inner1, int Inner2, int Outer,
                                            int Inner1Stride, int Inner2Stride, int OuterStride, int d, bool bnh) {


        int n = blockIdx.x;
        int c = blockIdx.y;
        int inner1 = blockIdx.z;
        int inner2 = threadIdx.x;

        int temploc = ((((outer*2+d)*N+n)*outC+c)*Inner1+inner1)*Inner2+inner2;
        int tempm1loc = (((((outer-1+2*d)*2+d)*N+n)*outC+c)*Inner1+inner1)*Inner2+inner2;
        int hloc = (n*outC+c)*Outer*Inner1*Inner2+outer*OuterStride+inner1*Inner1Stride+inner2*Inner2Stride;
        int hdloctm1 = ((((d*N+n)*outC+c)*Outer+(outer-1+2*d))*Inner1+inner1)*Inner2+inner2;

        //mdrnn_bicgrus_backwardprepare_same(zd,htilded,deltahd,deltah,hd,rd,deltard,temploc,tempm1loc,hloc,hdloctm1,preouter);
        mdrnn_bicgrus_backwardprepare_same(zd[temploc], htilded[temploc], deltahd[temploc], deltahd[tempm1loc], deltah[hloc], hd[hdloctm1], rd[temploc], deltard[temploc], preouter, bnh);



    }


template <typename Dtype>
void CuDNNCGRU<Dtype>::Backward(const Dtype * deltah, const Dtype * x,Dtype * hd, Dtype * zd, Dtype * rd, Dtype * deltard, Dtype * htilded, Dtype * deltahd, const Dtype * fx, const Dtype * fh, const Dtype * b, Dtype * deltafx, Dtype * deltafh, Dtype * deltab, Dtype * deltax){
/*
    Dtype* hd = this->hd_->mutable_gpu_data();
    //Dtype* fx = weightsx->mutable_gpu_data();
    Dtype* fh = weightsh->mutable_gpu_data();
    Dtype * b = bias->mutable_gpu_data();

    Dtype* zd = this->zd_->mutable_gpu_data();
    Dtype* rd = this->rd_->mutable_gpu_data();
    Dtype* deltard = this->rd_->mutable_gpu_diff();
    Dtype* htilded = this->htilded_->mutable_gpu_data();

    Dtype* deltahd = this->htilded_->mutable_gpu_diff(); //we will use htilded as diff.
    //Dtype* deltahdbefore = this->zd_->mutable_gpu_diff();
    Dtype* deltah = top[0]->mutable_gpu_diff();
    Dtype* deltax = bottom[0]->mutable_gpu_diff();
    Dtype* x = bottom[0]->mutable_gpu_data();
    //Dtype* deltard = this->rd_->mutable_gpu_diff();
    //Dtype* deltaxd = this->xd_->mutable_gpu_diff();
    Dtype * xd;
    Dtype *xdatOuter;
//    Dtype manx;
//    if (this->bnx_) {
//        xd = this->xd_->mutable_gpu_data();
//        manx = this->bn_moving_meanvar_number_x_->mutable_cpu_data()[0];
//    }
    Dtype* deltafx = weightsx->mutable_gpu_diff();
    Dtype* deltafh = weightsh->mutable_gpu_diff();
    Dtype* deltab = bias->mutable_gpu_diff();
    Dtype * fxtmp = weightsx->mutable_gpu_data();
//    if (this->dropconnectx_ > 0) {
//        fxtmp = this->dropconnectxweights_->mutable_gpu_data();
//    }
    Dtype * fx = fxtmp;*/

    cudaMemset(deltafx,0,this->Fsy_*this->Fsz_*3*2*this->outC_*sizeof(Dtype));
    cudaMemset(deltafh,0,this->Fsy_*this->Fsz_*3*2*this->outC_*sizeof(Dtype));
    cudaMemset(deltab,0,3*2*this->outC_*sizeof(Dtype));
    cudaMemset(deltax,0,this->N_*this->inC_*this->X_*this->Y_*this->Z_*sizeof(Dtype));

    if (this->favorspeedovermemory_) {
        cudaMemset(deltahd,0,this->N_*this->outC_*this->X_*this->Y_*this->Z_*2*sizeof(Dtype));        
    } else {
        cudaMemset(deltahd,0,this->N_*this->outC_*this->Inner1_*this->Inner2_*2*sizeof(Dtype));
    }


    //gpuErrchk( cudaDeviceSynchronize() );

    const int NCoYZForTemps = this->N_*this->outC_*this->Inner1_*this->Inner2_;
    const int NCoXYZ = this->N_*this->outC_*this->Outer_*this->Inner1_*this->Inner2_;//this->OuterStride_*NCoYZ;
    const int CoCiFsyFsz = this->outC_*this->inC_*this->Fsy_*this->Fsz_;
    const int CoCoFsyFsz = this->outC_*this->outC_*this->Fsy_*this->Fsz_;

    for (int preouter = this->Outer_-1; preouter >= 0; preouter--) { //loop over 1 particular cgru direction.
        for (int d = 0; d < 2; d++) { //this needs to be inside preouter, since we want to keep deltahd zero for both directions in th ebeginning.
            int outer = preouter;
            if (d==1) {
                outer = this->Outer_-1-preouter;
            }
            int tempoffset = 0;
            int tempoffsetpast = 0;
            if (this->favorspeedovermemory_) {
                tempoffset = (outer*2+d)*NCoYZForTemps;
                tempoffsetpast = ((outer-1+2*d)*2+d)*NCoYZForTemps;
            } else {
                tempoffset = d*NCoYZForTemps;
                tempoffsetpast = tempoffset;
            }

            //gpuErrchk( cudaDeviceSynchronize() );
            //gpuErrchk( cudaPeekAtLastError() );
            /*if (this->bnx_) {
                xdatOuter = xd+d*this->Inner1_*this->Inner2_*this->inC_*this->N_;
                dim3 blocks(this->inC_);
                dim3 threads(128, 1, 1);

                Dtype * varx = this->bn_variance_x_->mutable_gpu_data()+(d*this->Outer_+outer)*this->inC_;
                if (this->bn_global_stats_) {
                    const Dtype * ravgmean = this->bn_moving_meanvar_x_->gpu_data()+d*2*this->inC_;
                    const Dtype * ravgvar = this->bn_moving_meanvar_x_->gpu_data()+(d*2+1)*this->inC_;
                    normalize_by_mean_and_variance_x_slice<<<blocks, threads>>>(x, xdatOuter, varx, ravgmean,ravgvar,
                                                                                manx, this->bn_epsilon_x_, outer, this->Outer_, this->Inner1_, this->Inner2_,
                                                                                this->OuterStride_, this->Inner1Stride_, this->Inner2Stride_, this->N_, this->inC_);
                } else {
                    const Dtype * mymean = this->bn_mean_x_->gpu_data()+(d*this->Outer_+outer)*this->inC_;
                    const Dtype * myvar = this->bn_variance_x_->gpu_data()+(d*this->Outer_+outer)*this->inC_;
                    normalize_by_mean_and_variance_x_slice_correct_values<<<blocks, threads>>>(x, xdatOuter, mymean,myvar,
                                                                                               outer, this->Outer_, this->Inner1_, this->Inner2_,
                                                                                               this->OuterStride_, this->Inner1Stride_, this->Inner2Stride_, this->N_, this->inC_);
                }


                gpuErrchk(cudaDeviceSynchronize());
                gpuErrchk( cudaPeekAtLastError() );
            } else {*/
                const Dtype * xdatOuter = x+outer*this->OuterStride_;
//            }

            dim3 blocks(this->N_, this->outC_, this->Inner1_);
            dim3 threads(this->Inner2_, 1, 1);
            if (this->favorspeedovermemory_) {
                mdrnn_bicgrus_backwardprepare_fast<<<blocks,threads>>>(zd,htilded,deltahd,deltah,hd,rd,deltard,
                                                             outer,preouter, this->N_, this->outC_,
                                                             this->Inner1_, this->Inner2_, this->Outer_,
                                                             this->Inner1Stride_, this->Inner2Stride_, this->OuterStride_,d,this->bnh_);
            } else {


                doForwardConvolutionsOneDirectionForZdRdHtilded_gpu(zd,rd,htilded,deltard,x+outer*this->OuterStride_,hd,fx,fh,b, outer, preouter, d);
                gpuErrchk( cudaDeviceSynchronize() );
                mdrnn_bicgrus_backwardprepare<<<blocks,threads>>>(zd,htilded,deltahd,deltah,hd,rd,deltard,
                                                             outer,preouter, this->N_, this->outC_,
                                                             this->Inner1_, this->Inner2_, this->Outer_,
                                                             this->Inner1Stride_, this->Inner2Stride_, this->OuterStride_,d,this->bnh_);
            }
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize());


            if (preouter > 0) {
                checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, bottomHTensor_,
                                                          hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_, this->topTempTensor_,  zd+tempoffset, convHDesc_,
                                                          backwdHf_, workspace_[d*3], this->workspaceSize_,
                                                          cudnn::dataType<Dtype>::one, filterHDesc_, deltafh+(d*3)*CoCoFsyFsz));
                checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3+1], cudnn::dataType<Dtype>::one, bottomHTensor_,
                                                          hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_, this->topTempTensor_,  deltard+tempoffset, convHDesc_,
                                                          backwdHf_, workspace_[d*3+1], this->workspaceSize_,
                                                          cudnn::dataType<Dtype>::one, filterHDesc_, deltafh+(d*3+1)*CoCoFsyFsz));
                checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3+2], cudnn::dataType<Dtype>::one, bottomHTensor_,
                                                          hd+d*NCoXYZ+(outer-1+2*d)*this->Inner1_*this->Inner2_, this->topTempTensor_,  rd+tempoffset, convHDesc_,
                                                          backwdHf_, workspace_[d*3+2], this->workspaceSize_,
                                                          cudnn::dataType<Dtype>::one, filterHDesc_, deltafh+(d*3+2)*CoCoFsyFsz));
                gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
                gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+1]));
                gpuErrchk( cudaStreamSynchronize(this->stream_[d*3+2]));
                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, filterHDesc_,
                                                        fh+(d*3)*CoCoFsyFsz, this->topTempTensor_, zd+tempoffset, convHDesc_,
                                                        backwdHd_,  workspace_[d*3], this->workspaceSize_,
                                                        cudnn::dataType<Dtype>::one, this->topTempTensor_, deltahd+tempoffsetpast));
                gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, filterHDesc_,
                                                        fh+(d*3+1)*CoCoFsyFsz, this->topTempTensor_, deltard+tempoffset, convHDesc_,
                                                        backwdHd_,  workspace_[d*3], this->workspaceSize_,
                                                        cudnn::dataType<Dtype>::one, this->topTempTensor_, deltahd+tempoffsetpast));
                gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, filterHDesc_,
                                                        fh+(d*3+2)*CoCoFsyFsz, this->topTempTensor_, rd+tempoffset, convHDesc_,
                                                        backwdHd_,  workspace_[d*3], this->workspaceSize_,
                                                        cudnn::dataType<Dtype>::one, this->topTempTensor_, deltahd+tempoffsetpast));
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
                //gpuErrchk( cudaStreamSynchronize(this->stream_[d*3]));
            }


            checkCUDNN(cudnnConvolutionBackwardBias(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, topTempTensor_,
                                                    zd+tempoffset,  cudnn::dataType<Dtype>::one, biasTensor_, deltab+(d*3)*this->outC_));
            checkCUDNN(cudnnConvolutionBackwardBias(this->cudnnHandle_[d*3+1], cudnn::dataType<Dtype>::one, topTempTensor_,
                                                    deltard+tempoffset,  cudnn::dataType<Dtype>::one, biasTensor_, deltab+(d*3+1)*this->outC_));
            checkCUDNN(cudnnConvolutionBackwardBias(this->cudnnHandle_[d*3+2], cudnn::dataType<Dtype>::one, topTempTensor_,
                                                    htilded+tempoffset,  cudnn::dataType<Dtype>::one, biasTensor_, deltab+(d*3+2)*this->outC_));




        checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3], cudnn::dataType<Dtype>::one, bottomXTensor_,
                                                  xdatOuter, topTempTensor_,  zd+tempoffset, convXDesc_,
                                                  backwdXf_, workspaceN_[d*3], this->workspaceSizeN_,
                                                  cudnn::dataType<Dtype>::one, filterXDesc_, deltafx+(d*3)*CoCiFsyFsz));
        checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3+1], cudnn::dataType<Dtype>::one, bottomXTensor_,
                                                  xdatOuter, topTempTensor_,  deltard+tempoffset, convXDesc_,
                                                  backwdXf_, workspaceN_[d*3+1], this->workspaceSizeN_,
                                                  cudnn::dataType<Dtype>::one, filterXDesc_, deltafx+(d*3+1)*CoCiFsyFsz));
        checkCUDNN(cudnnConvolutionBackwardFilter(this->cudnnHandle_[d*3+2], cudnn::dataType<Dtype>::one, bottomXTensor_,
                                                  xdatOuter, topTempTensor_,  htilded+tempoffset, convXDesc_,
                                                  backwdXf_, workspaceN_[d*3+2], this->workspaceSizeN_,
                                                  cudnn::dataType<Dtype>::one, filterXDesc_, deltafx+(d*3+2)*CoCiFsyFsz));

            //if (propagate_down) {
                gpuErrchk( cudaDeviceSynchronize() );
                gpuErrchk( cudaPeekAtLastError() );
                Dtype * deltaxAtOuter = deltax+outer*this->OuterStride_;
//                if (this->bnx_) {
//                    deltaxAtOuter = this->xd_->mutable_gpu_diff()+d*this->N_*this->inC_*this->Inner1_*this->Inner2_;
//                }

                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[0], cudnn::dataType<Dtype>::one, filterXDesc_,
                                                        fx+(d*3)*CoCiFsyFsz, topTempTensor_, zd+tempoffset, convXDesc_,
                                                        backwdXd_,  workspaceN_[0], this->workspaceSizeN_,
                                                        this->bnx_ ? cudnn::dataType<Dtype>::zero : cudnn::dataType<Dtype>::one , bottomXTensor_, deltaxAtOuter));
                gpuErrchk( cudaStreamSynchronize(this->stream_[0]));
                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[0], cudnn::dataType<Dtype>::one, filterXDesc_,
                                                        fx+(d*3+1)*CoCiFsyFsz, topTempTensor_, deltard+tempoffset, convXDesc_,
                                                        backwdXd_,  workspaceN_[0], this->workspaceSizeN_,
                                                        cudnn::dataType<Dtype>::one , bottomXTensor_, deltaxAtOuter));
                gpuErrchk( cudaStreamSynchronize(this->stream_[0]));
                checkCUDNN(cudnnConvolutionBackwardData(this->cudnnHandle_[0], cudnn::dataType<Dtype>::one, filterXDesc_,
                                                        fx+(d*3+2)*CoCiFsyFsz, topTempTensor_, htilded+tempoffset, convXDesc_,
                                                        backwdXd_,  workspaceN_[0], this->workspaceSizeN_,
                                                        cudnn::dataType<Dtype>::one , bottomXTensor_, deltaxAtOuter));
                gpuErrchk( cudaStreamSynchronize(this->stream_[0]));
                /*if (this->bnx_) {

                    dim3 blocks(this->inC_);
                    dim3 threads(128, 1, 1);
                    Dtype * varx = this->bn_variance_x_->mutable_gpu_data()+(d*this->Outer_+outer)*this->inC_;
                    backpropagate_bn_x_slice<<<blocks, threads>>>(deltax, xdatOuter, varx, deltaxAtOuter, outer, this->Outer_, this->Inner1_, this->Inner2_,
                                                                                    this->OuterStride_, this->Inner1Stride_, this->Inner2Stride_, this->N_, this->inC_);
                }*/
//            }
            gpuErrchk( cudaDeviceSynchronize() );
            gpuErrchk( cudaPeekAtLastError() );
        }
    }
    /*if (this->dropconnectx_ > 0) {
        float * filtermask = this->dropconnectxmask_->mutable_gpu_data();
        dim3 blocks(this->inC_* this->outC_* 2*3);
        dim3 threads(this->Fsy_* this->Fsz_);
        mask<<<blocks,threads>>>(filtermask,deltafx);

    }*/
}

template class CuDNNCGRU<float>;


#endif
