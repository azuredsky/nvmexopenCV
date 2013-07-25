#include<mex.h>
#include<stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/gpumat.hpp"
#include "opencv2/core/core.hpp"
#include <cuda.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

 void mexFunction(int nlhs,              /* number of expected outputs */
                 mxArray* plhs[],       /* output pointer array */
                 int nrhs,              /* number of inputs */
                 const mxArray* prhs[]  /* input pointer array */ )
{      
      // Check arguments
    if (nlhs!=0 || nrhs!=0)
        mexErrMsgIdAndTxt("myfunc:invalidArgs", "Wrong number of arguments");
    
      printf("%d\n",getCudaEnabledDeviceCount());
        
      cv::Mat src_host = cv::imread("3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
      cv::gpu::GpuMat dst, src;
      src.upload(src_host);

      cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
              
      cv::imshow("Src", src_host);
      
      cv::Mat result_host(dst);
      cv::imshow("Result", result_host);
      cv::waitKey();
 }