/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer. 

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution. 

The name of the author may not be used to endorse or promote products
derived from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Copyright 2008 Yuancheng Luo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <cutil.h>

#include "mex.h"
#include "matrix.h"

#include "cuda.h"


//Definitions
#define GAUSS_WIN_RADI	7
#define GAUSS_WIN_WIDTH	(GAUSS_WIN_RADI*2+1)

#define BLOCK_WIDTH		16
#define BLOCK_HEIGHT	16
#define SOBEL_SMEM_WIDTH	 18
#define SOBEL_SMEM_HEIGHT	 18

#define HYST_MAX_SIZE	128


//Prototypes

__global__ void loadConvertData(unsigned char *iData,float *formatData,
								unsigned short int iWidth,unsigned short int iHeight);
__global__ void loadConvertData8(unsigned char *iData,float *formatData,
								unsigned short int iWidth,unsigned short int iHeight);
__global__ void gaussianSeparablePassX(	float *formatData,float *xPassData,
									    unsigned short int iWidth,unsigned short int iHeight,
										float *kernel);
__global__ void gaussianSeparablePassY(float *xPassData, float *oData,
									    unsigned short int iWidth,unsigned short int iHeight,
										float *kernel);

__global__ void sobelSeparablePassX(float  *formatData,float *xPassData,
						   unsigned short int iWidth,unsigned short int iHeight,
						   float kernel[3]);
__global__ void sobelSeparablePassY(float *xPassData,float *oData,
						   unsigned short int iWidth,unsigned short int iHeight,
						   float kernel[3]);

__global__ void cannyGradientStrengthDirection(float *gradX,float *gradY,
												unsigned short int iWidth, unsigned short int iHeight,
												float *gradStrength,unsigned int *gradDirection);
__global__ void cannyNonmaxSupression(	unsigned int *gradDirection,float *gradStrength,float *gradStrengthOut,
										unsigned short int iWidth,unsigned short int iHeight);
__global__ void cannyHysteresisBlock(	float *gradStrength,float *gradStrengthOut,
										unsigned short iWidth,unsigned short iHeight,
										float thresholdLow, float thresholdHigh);
__global__ void cannyHysteresisBlockShared(	float *gradStrength,float *gradStrengthOut,
											unsigned short iWidth,unsigned short iHeight,
											float thresholdLow, float thresholdHigh);

__global__ void cannyBlockConverter(float *gradStrength,
									unsigned char *outputImage,
									unsigned short iWidth,unsigned short iHeight);
__global__ void cannyBlockConverter8(float *gradStrength,
									unsigned char *outputImage,
									unsigned short iWidth,unsigned short iHeight);


//Globals
float *gradientStrength,*gradientStrengthOut;
unsigned int *gradientDirection;
float *gaussKernelX, *gaussKernelY;
float *kernelXGradX,*kernelYGradX,*kernelXGradY,*kernelYGradY;
float *formatData,*xPassData;
float *gradX,*gradY;
unsigned char *inputImage;
unsigned char *outputImage;

float sobelKxgradx[3]={-1,0,1};
float sobelKygradx[3]={1,2,1};

float sobelKxgrady[3]={1,2,1};
float sobelKygrady[3]={1,0,-1};


void generateGaussiankernels(float sigma){
	float xK[GAUSS_WIN_WIDTH],yK[GAUSS_WIN_WIDTH];
	float sumX=0,sumY=0;
	for(int a=-GAUSS_WIN_RADI;a<=GAUSS_WIN_RADI;++a){
		yK[a+GAUSS_WIN_RADI]=exp(-a*a/(2*sigma*sigma));
		xK[a+GAUSS_WIN_RADI]=(1.0/(6.283185307179586*sigma*sigma))*yK[a+GAUSS_WIN_RADI];
		
		sumX+=xK[a+GAUSS_WIN_RADI];
		sumY+=yK[a+GAUSS_WIN_RADI];
	}

	//Normalize
	for(int a=0;a<GAUSS_WIN_WIDTH;++a){
		xK[a]/=sumX;
		//printf("%f ",xK[a]);	
	}
	printf("\n");
	for(int a=0;a<GAUSS_WIN_WIDTH;++a){
		yK[a]/=sumY;
		//printf("%f ",yK[a]);	
	}

	cudaMemcpy(gaussKernelX,xK,GAUSS_WIN_WIDTH*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(gaussKernelY,yK,GAUSS_WIN_WIDTH*sizeof(float),cudaMemcpyHostToDevice);
}

void cannyInit(int iWidth,int iHeight,float sigma){

	cudaSetDevice(0);

	int kernelSize=sizeof(float)*3;
	cudaMalloc((void**)&kernelXGradX,kernelSize);
	cudaMalloc((void**)&kernelYGradX,kernelSize);
	cudaMalloc((void**)&kernelXGradY,kernelSize);
	cudaMalloc((void**)&kernelYGradY,kernelSize);

	cudaMalloc((void**)&formatData,iWidth*iHeight*sizeof(float));
	cudaMalloc((void**)&xPassData,iWidth*iHeight*sizeof(float));
	cudaMalloc((void**)&gradX,iWidth*iHeight*sizeof(float));
	cudaMalloc((void**)&gradY,iWidth*iHeight*sizeof(float));

	cudaMemcpy(kernelXGradX,sobelKxgradx,kernelSize,cudaMemcpyHostToDevice);
	cudaMemcpy(kernelYGradX,sobelKygradx,kernelSize,cudaMemcpyHostToDevice);
	cudaMemcpy(kernelXGradY,sobelKxgrady,kernelSize,cudaMemcpyHostToDevice);
	cudaMemcpy(kernelYGradY,sobelKygrady,kernelSize,cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&gradientStrength,iWidth*iHeight*sizeof(float));
	cudaMalloc((void**)&gradientStrengthOut,iWidth*iHeight*sizeof(float));
	cudaMalloc((void**)&gradientDirection,iWidth*iHeight*sizeof(int));

	cudaMalloc((void**)&inputImage,iWidth*iHeight*sizeof(char)*3);
	cudaMalloc((void**)&outputImage,iWidth*iHeight*sizeof(char)*3);

	cudaMalloc((void**)&gaussKernelX,GAUSS_WIN_WIDTH*sizeof(float));
	cudaMalloc((void**)&gaussKernelY,GAUSS_WIN_WIDTH*sizeof(float));

	if(sigma>0) generateGaussiankernels(sigma);
}

void cannyFree(){
	cudaFree(kernelXGradX);
	cudaFree(kernelYGradX);
	cudaFree(kernelXGradY);
	cudaFree(kernelYGradY);

	cudaFree(formatData);
	cudaFree(xPassData);
	cudaFree(gradX);
	cudaFree(gradY);

	cudaFree(gradientStrength);
	cudaFree(gradientStrengthOut);
	cudaFree(gradientDirection);

	cudaFree(inputImage);
	cudaFree(outputImage);
	
	cudaFree(gaussKernelX);
	cudaFree(gaussKernelY);

}

/*
Test matlab column-major format:
*/
__global__ void matlabGrayScaleTest(unsigned char *in_data,unsigned char *out_data,
									unsigned short int iWidth,unsigned short int iHeight){
	
	__shared__ int smem[3][BLOCK_WIDTH][BLOCK_HEIGHT];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;

	//Coalesced Read Test
	unsigned char txMod=tx-(tx/4*4);

	unsigned int globalMemAddress=(tx/4)*(iWidth*iHeight)+(iHeight*BLOCK_WIDTH*bx+BLOCK_HEIGHT*by)+iHeight*ty+txMod*4;//Every halfwarp should be multiple of 16*sizeof(type)
	unsigned short int halfwarpSmemAddressPlusOffset=(tx/4)*(BLOCK_WIDTH*BLOCK_HEIGHT)+txMod*4;

	if(tx<12){//As long as every halfwarp thread is aligned and subsequent threads read from halfwarpthread+1 address of type with size 4, coalesce will occur
		unsigned int iDataVal=*((int*)(in_data+globalMemAddress));//Single 4byte read
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset)=(iDataVal)&0x000000FF;//Read into integer smem(no bank conflicts)
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+1)=(iDataVal>>8)&0x000000FF;
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+2)=(iDataVal>>16)&0x000000FF;
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+3)=(iDataVal>>24)&0x000000FF;
	}
	
	
    __syncthreads();
	unsigned char gray=smem[0][tx][ty]*.11+smem[1][tx][ty]*.59+smem[2][tx][ty]*.3;
	

	//Coalesced Write Test
	__shared__ unsigned int sharedStrength[3][BLOCK_WIDTH][BLOCK_HEIGHT];
	//Load center
	__syncthreads();
	sharedStrength[0][tx][ty]=gray;
	sharedStrength[1][tx][ty]=gray;
	sharedStrength[2][tx][ty]=gray;
	

	x=(tx/4)*(BLOCK_WIDTH*BLOCK_HEIGHT)+txMod*4;
	__syncthreads();
	if(tx<12){
		*((unsigned int *)(out_data+globalMemAddress))=\
			   *(sharedStrength[0][ty]+x+0)\
			+((*(sharedStrength[0][ty]+x+1))<<8)\
			+((*(sharedStrength[0][ty]+x+2))<<16)\
			+((*(sharedStrength[0][ty]+x+3))<<24);
	}


}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){
	
	unsigned char *in_data;
	unsigned char *out_data;
	float thresholdLow,thresholdHigh;
	float sigma;
	int hysteresisIts;

	int nChannels;
	int iWidth,iHeight;

	nChannels=mxGetScalar(prhs[1]);

	iHeight=mxGetM(prhs[0]);
	iWidth=mxGetN(prhs[0])/nChannels;
	
	//Setup CUDA input/output vars
	cudaMalloc((void**)&in_data,iWidth*iHeight*3*sizeof(unsigned char));
	cudaMalloc((void**)&out_data,iWidth*iHeight*3*sizeof(unsigned char));

	if(nChannels==3)		cudaMemcpy(in_data,(unsigned char *)mxGetData(prhs[0]),iWidth*iHeight*3*sizeof(char),cudaMemcpyHostToDevice);
	else if(nChannels==1)	cudaMemcpy(in_data,(unsigned char *)mxGetData(prhs[0]),iWidth*iHeight*sizeof(char),cudaMemcpyHostToDevice);

	thresholdLow=mxGetScalar(prhs[2]);
	thresholdHigh=mxGetScalar(prhs[3]);
	hysteresisIts=mxGetScalar(prhs[4]);
	sigma=mxGetScalar(prhs[5]);

	//Create Output
	const mwSize dims[3]={iHeight,iWidth,3};
	//mwSize dims[3]={iHeight,iWidth,3};
	plhs[0]=mxCreateNumericArray (3,dims,mxUINT8_CLASS,mxREAL);
	

	mexPrintf("matrix dims %d %d\n thresholdlow %f, thresholdhigh %f,hysteresisIts %d\n",iHeight,iWidth,thresholdLow,thresholdHigh,hysteresisIts);



	///////////////Cuda Part///////////////
	cannyInit(iWidth,iHeight,sigma);

	dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	dim3 grid((iWidth/(float)block.x),(iHeight/(float)block.y), 1);
	
//	matlabGrayScaleTest<<<grid,block>>>(in_data,out_data,iWidth,iHeight);

	//Get data
	if(nChannels==3)		loadConvertData<<<grid,block>>>(in_data,formatData,iWidth,iHeight);
	else if(nChannels==1)	loadConvertData8<<<grid,block>>>(in_data,formatData,iWidth,iHeight);

	//Gaussian	
	if(sigma>0){
		gaussianSeparablePassX<<<grid,block>>>(formatData,xPassData,iWidth,iHeight,gaussKernelX);
		gaussianSeparablePassY<<<grid,block>>>(xPassData,formatData,iWidth,iHeight,gaussKernelY);
	}
	//xgradient	
	sobelSeparablePassX<<<grid,block>>>(formatData,xPassData,iWidth,iHeight,kernelXGradX);
	sobelSeparablePassY<<<grid,block>>>(xPassData,gradX,iWidth,iHeight,kernelYGradX);


	//ygradient
	sobelSeparablePassX<<<grid,block>>>(formatData,xPassData,iWidth,iHeight,kernelXGradY);
	sobelSeparablePassY<<<grid,block>>>(xPassData,gradY,iWidth,iHeight,kernelYGradY);



	//Find magnitude and direction
	cannyGradientStrengthDirection<<<grid,block>>>(	gradX,gradY,
													iWidth,iHeight,
													gradientStrength,gradientDirection);

	//Find nonmaximum supression (thin pixels)
	cannyNonmaxSupression<<<grid,block>>>(	gradientDirection,gradientStrength,gradientStrengthOut,
											iWidth,iHeight);
	
	//Use Hysteresis (growing BLOCK method) to elimate streaking
	for(int a=0;a<hysteresisIts;++a){
		
		
		//Slow local memory intensive version
	//	cannyHysteresisBlock<<<grid,block>>>(	gradientStrengthOut,gradientStrength,
	//											iWidth,iHeight,
	//											thresholdLow, thresholdHigh);
									

		//Fast version using shared memory only and parallel reduction
		cannyHysteresisBlockShared<<<grid,block>>>(	gradientStrengthOut,gradientStrength,
												iWidth,iHeight,
												thresholdLow, thresholdHigh);
										
		cudaMemcpy(gradientStrengthOut,gradientStrength,iWidth*iHeight*sizeof(float),cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
	}

	if(nChannels==3) {
		cannyBlockConverter<<<grid,block>>>(gradientStrength,out_data,iWidth,iHeight);
		cudaMemcpy((unsigned char *)mxGetData(plhs[0]),out_data,iWidth*iHeight*3*sizeof(char),cudaMemcpyDeviceToHost);
	}
	else if(nChannels==1){
		cannyBlockConverter8<<<grid,block>>>(gradientStrength,out_data,iWidth,iHeight);
		cudaMemcpy((unsigned char *)mxGetData(plhs[0]),out_data,iWidth*iHeight*sizeof(char),cudaMemcpyDeviceToHost);
	}


	
	cannyFree();

	
}

__global__ void loadConvertData(unsigned char *iData,float *formatData,
								unsigned short int iWidth,unsigned short int iHeight){
	__shared__ int smem[3][BLOCK_WIDTH][BLOCK_HEIGHT];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;

	//Matlab version
	unsigned int ref=iHeight*x+y;
	unsigned char txMod=tx-(tx/4*4);

	unsigned int globalMemAddress=(tx/4)*(iWidth*iHeight)+(iHeight*BLOCK_WIDTH*bx+BLOCK_HEIGHT*by)+iHeight*ty+txMod*4;//Every halfwarp should be multiple of 16*sizeof(type)
	unsigned short int halfwarpSmemAddressPlusOffset=(tx/4)*(BLOCK_WIDTH*BLOCK_HEIGHT)+txMod*4;

	if(tx<12){//As long as every halfwarp thread is aligned and subsequent threads read from halfwarpthread+1 address of type with size 4, coalesce will occur
		unsigned int iDataVal=*((int*)(iData+globalMemAddress));//Single 4byte read
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset)=(iDataVal)&0x000000FF;//Read into integer smem(no bank conflicts)
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+1)=(iDataVal>>8)&0x000000FF;
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+2)=(iDataVal>>16)&0x000000FF;
		*(&smem[0][ty][0]+halfwarpSmemAddressPlusOffset+3)=(iDataVal>>24)&0x000000FF;
	}
	
    __syncthreads();
	unsigned char gray=smem[0][tx][ty]*.11+smem[1][tx][ty]*.59+smem[2][tx][ty]*.3;

	ref=iWidth*y+x;
	formatData[ref]=gray;
}


__global__ void loadConvertData8(unsigned char *iData,float *formatData,
								unsigned short int iWidth,unsigned short int iHeight){
	__shared__ int smem[BLOCK_WIDTH][BLOCK_HEIGHT];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;

	//Matlab version
	unsigned int ref=iHeight*x+y;

	unsigned int globalMemAddress=(iHeight*BLOCK_WIDTH*bx+BLOCK_HEIGHT*by)+iHeight*ty+tx*4;//Every halfwarp should be multiple of 16*sizeof(type)
	unsigned short int halfwarpSmemAddressPlusOffset=tx*4;

	if(tx<4){//As long as every halfwarp thread is aligned and subsequent threads read from halfwarpthread+1 address of type with size 4, coalesce will occur
		unsigned int iDataVal=*((int*)(iData+globalMemAddress));//Single 4byte read
		*(&smem[ty][0]+halfwarpSmemAddressPlusOffset)=(iDataVal)&0x000000FF;//Read into integer smem(no bank conflicts)
		*(&smem[ty][0]+halfwarpSmemAddressPlusOffset+1)=(iDataVal>>8)&0x000000FF;
		*(&smem[ty][0]+halfwarpSmemAddressPlusOffset+2)=(iDataVal>>16)&0x000000FF;
		*(&smem[ty][0]+halfwarpSmemAddressPlusOffset+3)=(iDataVal>>24)&0x000000FF;
	}
	
    __syncthreads();
	unsigned char gray=smem[tx][ty];

	ref=iWidth*y+x;
	formatData[ref]=gray;
}

__global__ void gaussianSeparablePassX(	float *formatData,float *xPassData,
									    unsigned short iWidth,unsigned short iHeight,
										float *kernel){
	__shared__ int smem[BLOCK_HEIGHT][BLOCK_WIDTH+2*GAUSS_WIN_RADI];

	__shared__ float k[GAUSS_WIN_WIDTH];			//Gaussian kernel data

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned short x=BLOCK_WIDTH*bx+tx;
	unsigned short y=BLOCK_HEIGHT*blockIdx.y+ty;
	unsigned int ref2=iWidth*y+x;					//Pixel address in image


	//Load Center
	smem[ty][GAUSS_WIN_RADI+tx]=formatData[ref2];
	
	//Read left block
	float s=(bx>0) ? (formatData[ref2-16]) :(0);
	__syncthreads();
	//Write left apron
	if(tx>=BLOCK_WIDTH-GAUSS_WIN_RADI)	smem[ty][tx-(BLOCK_WIDTH-GAUSS_WIN_RADI)]=s;
	__syncthreads();

	//Read/write right apon
	if(tx<GAUSS_WIN_RADI)		smem[ty][GAUSS_WIN_RADI+BLOCK_WIDTH+tx]=(x+BLOCK_WIDTH<iWidth) ? (formatData[ref2+BLOCK_WIDTH]) :( 0);
	__syncthreads();

	////Load 3word kernel data
	if(ty==0&&tx<GAUSS_WIN_WIDTH) k[tx]=kernel[tx];
	__syncthreads();

	////Convolve in X Direction
	float strength=0;
	#pragma unroll
	for(unsigned char a=0;a<GAUSS_WIN_WIDTH;++a)
		strength+=smem[ty][tx+a]*k[a];
   __syncthreads();
   xPassData[ref2]=strength;
}


__global__ void gaussianSeparablePassY(float *xPassData, float *oData,
									    unsigned short int iWidth,unsigned short int iHeight,
										float *kernel){
	__shared__ float smem[BLOCK_HEIGHT+2*GAUSS_WIN_RADI][BLOCK_WIDTH];
	__shared__ float k[GAUSS_WIN_WIDTH];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;
	unsigned int ref2=iWidth*y+x;

	//Load center
	smem[GAUSS_WIN_RADI+ty][tx]=xPassData[ref2];
	__syncthreads();
	
	if(ty<GAUSS_WIN_RADI){
		//Load top
		smem[ty][tx]=(by>0) ? (xPassData[ref2-iWidth*GAUSS_WIN_RADI]) :(0);
		//Load bottom
		smem[GAUSS_WIN_RADI+BLOCK_HEIGHT+ty][tx]=(y+BLOCK_HEIGHT<iHeight) ? (xPassData[ref2+iWidth*BLOCK_HEIGHT]) : (0);
	}
	__syncthreads();

	////Load 3word kernel data
	if(ty==0&&tx<GAUSS_WIN_WIDTH) k[tx]=kernel[tx];

	////Convolve in Y Direction
	float strength=0;
	__syncthreads();
	
	#pragma unroll
	for(unsigned char a=0;a<GAUSS_WIN_WIDTH;++a)
		strength+=smem[ty+a][tx]*k[a];

   __syncthreads();
   oData[ref2]=strength;
}
/*
iData: 3 channel (1byte per channel) input image
oData: intermediate 3 channel (1byte per channel) output image of results from first pass 
Description: Perform sobel separable convolution in x direction
*/
__global__ void sobelSeparablePassX(float *formatData,float *xPassData,
						   unsigned short int iWidth,unsigned short int iHeight,
						   float kernel[3]){
	__shared__ int smem[BLOCK_HEIGHT][SOBEL_SMEM_WIDTH];

	__shared__ float k[3];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;
	unsigned int ref2=iWidth*y+x;

	//Load Center
	smem[ty][tx+1]=formatData[ref2];
	
	////Load Apron
	if(tx==0){
		//Load left most column
		smem[ty][tx]=(x>0?formatData[ref2-1]:0);
		//Load right most column
		smem[ty][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth?formatData[ref2+BLOCK_WIDTH]:0);
	}

	////Load 3word kernel data
	__syncthreads();
	if(ty==0&&tx<3) k[tx]=kernel[tx];
	
   ////Convolve in X Direction
   //Loop unrolled and broadcast convolution kernel
	float strength;
	__syncthreads();
   strength=smem[ty][tx]*k[0]+smem[ty][tx+1]*k[1]+smem[ty][tx+2]*k[2];

   __syncthreads();
   xPassData[ref2]=strength;
}


/*
iData: 3 channel (1byte per channel) input image
oData: intermediate 3 channel (1byte per channel) output image of results from first pass 
Description: Perform sobel separable convolution in y direction
xPassData is in the format of channel(3)XheightXwidth
*/
__global__ void sobelSeparablePassY(float *xPassData,float *oData,
						   unsigned short int iWidth,unsigned short int iHeight,
						   float kernel[3]){
	__shared__ float smem[SOBEL_SMEM_HEIGHT][BLOCK_WIDTH];
	__shared__ float k[3];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned short int x=BLOCK_WIDTH*blockIdx.x+tx;
	unsigned short int y=BLOCK_HEIGHT*blockIdx.y+ty;
	unsigned int ref2=iWidth*y+x;

	////Load iData chunks into shared memory with coaeleased global reads
	//No need to load corners
	//Load center (implicit coalesce reads from data format)
	__syncthreads();
	smem[ty+1][tx]=xPassData[ref2];
	
	if(ty==0){
		//Load top (Implicit coalesce as center)
		smem[0][tx]=(y>0?xPassData[ref2-iWidth]:0);
		//Load bottom (Implicit coalesce as center)
		smem[SOBEL_SMEM_HEIGHT-1][tx]=(y+BLOCK_HEIGHT<iHeight-1?xPassData[ref2+iWidth*(BLOCK_HEIGHT+1)]:0);
	}

	////Load 3word kernel data
	__syncthreads();
	if(ty==0&&tx<3) k[tx]=kernel[tx];

   ////Convolve in Y Direction
	__syncthreads();
   float strength=(smem[ty][tx]*k[0]+smem[ty+1][tx]*k[1]+smem[ty+2][tx]*k[2]);

   // Divide by 4 to set range from 0-255
   __syncthreads();
   oData[ref2]=strength;

}


__device__ unsigned int angleToDirection(float y,float x){
	//Calculate direction

	float dir=((x==0&&y==0)?0:atan2(y,x));
	  
	if((dir>0.39269908169872414&&dir<=1.1780972450961724)||(dir>=-2.748893571891069&&dir<-1.9634954084936207)) return 45;
	else if((dir>1.1780972450961724&&dir<=1.9634954084936207)||(dir>=-1.9634954084936207&&dir<-1.1780972450961724)) return 90;
	else if((dir>1.9634954084936207&&dir<=2.748893571891069)||(dir>=-1.1780972450961724&&dir<-0.39269908169872414)) return 135;
	else return 0;
}

/*
calculate gradient magnitude and direction
*/
__global__ void cannyGradientStrengthDirection(float *gradX,float *gradY,
												unsigned short int iWidth, unsigned short int iHeight,
												float *gradStrength,unsigned int *gradDirection){
	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned short int x=BLOCK_WIDTH*blockIdx.x+tx;
	unsigned short int y=BLOCK_HEIGHT*blockIdx.y+ty;
    unsigned int ref2=iWidth*y+x;

	//get magnitude while loading to reuse space
	float strength,strength2,strengthMag;

	//Load gradX into shared mem
	//Load gradY into shared mem
	strength=gradX[ref2];
	strength2=gradY[ref2];

	//Calculate and coalesce write the most "salient" gradient magnitude to global mem
	strengthMag=sqrt(strength*strength+strength2*strength2);

	__syncthreads();
	gradStrength[ref2]=strengthMag;
	gradDirection[ref2]=angleToDirection(strength2,strength);

}


/*
4 direction vectors used to represent orientation and cover all 8 neighboring pixels.
	gradDirection={	0->left to right
					45->top left to bot right
					90->top to down
					135->top right to left
*/
__global__ void cannyNonmaxSupression(	unsigned int *gradDirection,float *gradStrength,float *gradStrengthOut,
										unsigned short int iWidth,unsigned short int iHeight){
	__shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
	unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
	unsigned int ref2=iWidth*y+x;
	
	////Load gradient strength data
	////And 1 pixel apron

	//Load center (implicit coalesce reads from data format)
	sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

	if(ty==0){//Load top (Implicit coalesce as center)
		sharedGradStrength[0][tx+1]=(y>0?gradStrength[ref2-iWidth]:0);
	}
	if(ty==BLOCK_HEIGHT-1){//Load bottom
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y<iHeight-1?gradStrength[ref2+iWidth]:0);
	}

	if(tx==0){
		//Load leftmost column (uncoalesced but only 1 thread per halfwarp)
		sharedGradStrength[ty+1][0]=(x>0?gradStrength[ref2-1]:0);
		//Load rightmost column (coalesced but only 1 thread per halfwar)
		sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth?gradStrength[ref2+BLOCK_WIDTH]:0);
	}

	//Corners
	if(tx==0&&ty==0){
		sharedGradStrength[0][0]=(x>0&&y>0?gradStrength[ref2-(iWidth+1)]:0);//TL
	}else if(tx==BLOCK_WIDTH-1&&ty==0){
		sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y>0?gradStrength[ref2-(iWidth-1)]:0);//TR
	}else if(tx==0&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y<iHeight-1?gradStrength[ref2+(iWidth-1)]:0);//BL
	}else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x<iWidth-1&&y<iHeight-1?gradStrength[ref2+(iWidth+1)]:0);//BR
	}

	__syncthreads();

	unsigned int f=gradDirection[ref2];
	
	x=(f==135?-1:(f==90?0:1));
	y=(f==135||f==45?-1:(f==0?0:1));

	//Is thread a maximum? //High chance of bank conflict
	bool a=(sharedGradStrength[ty+1][tx+1]>max(sharedGradStrength[ty+1+y][tx+1+x],sharedGradStrength[ty+1-y][tx+1-x]));
	
	//Suppress gradient in all nonmaximum pixels
	__syncthreads();
	gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1]*a;

}


__global__ void cannyHysteresisBlock(	float *gradStrength,float *gradStrengthOut,
										unsigned short int iWidth,unsigned short int iHeight,
										float thresholdLow, float thresholdHigh){
	__shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
	unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
    
	
	unsigned int ref2=iWidth*y+x;
	
	//Load center
	sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];

	if(ty==0){//Load top
		sharedGradStrength[0][tx+1]=(y>0) ? (gradStrength[ref2-iWidth]): (0);
	}else if(ty==BLOCK_HEIGHT-1){//Load bottom
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+iWidth]): (0);
	}

	__syncthreads();

	if(tx==0){//Load left 
		sharedGradStrength[ty+1][0]=(x>0) ? (gradStrength[ref2-1]) : (0);
	}else if(tx==BLOCK_WIDTH-1){//Load right
		sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth) ? (gradStrength[ref2+1]) : (0);
	}

	__syncthreads();

	//Corners
	if(tx==0&&ty==0){
		sharedGradStrength[0][0]=(x>0&&y>0) ? (gradStrength[ref2-(iWidth+1)]) : (0);//TL
	}else if(tx==BLOCK_WIDTH-1&&ty==0){
		sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y>0 ) ? (gradStrength[ref2-(iWidth-1)]) : (0);//TR
	}else if(tx==0&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+(iWidth-1)]) : (0);//BL
	}else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y+BLOCK_HEIGHT<iHeight)? (gradStrength[ref2+(iWidth+1)]) : (0);//BR
	}
	
	__syncthreads();
	//Initialization part
	//Check if neighbors are edge pixels
	float str=sharedGradStrength[ty+1][tx+1];
	__syncthreads();

	if(str>thresholdHigh)		str=-2;
	else if(str>thresholdLow)	str=-1;
	else if(str>0)				str=0;

	sharedGradStrength[ty+1][tx+1]=str;

	__syncthreads();
	
	unsigned char list[HYST_MAX_SIZE][2];	//Dump into Local memory
	unsigned short listOff=0;

	if(str==-1){
		//Search neighbors
		//Seed list
		if(	sharedGradStrength[ty][tx]==-2||sharedGradStrength[ty][tx+1]==-2||sharedGradStrength[ty][tx+2]==-2||
			sharedGradStrength[ty+1][tx]==-2||sharedGradStrength[ty+1][tx+2]==-2||
			sharedGradStrength[ty+2][tx]==-2||sharedGradStrength[ty+2][tx+1]==-2||sharedGradStrength[ty+2][tx+2]==-2){

			list[listOff][0]=ty+1;
			list[listOff++][1]=tx+1;
		}
	}

	unsigned char txReplace,tyReplace;
	__syncthreads();
	

	//Grow an edge and set potential edges
	for(x=0;x<listOff;++x){//While potential edge
		
		ty=list[x][0];
		tx=list[x][1];

		sharedGradStrength[ty][tx]=-2;//Set as definite edge
		
		//Check neighbors
		if(listOff<HYST_MAX_SIZE){
			if(sharedGradStrength[ty][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
				list[listOff][0]=ty;
				list[listOff++][1]=txReplace;
			}
			
			if(sharedGradStrength[ty][txReplace=max(tx-1,0)]==-1){
				list[listOff][0]=ty;
				list[listOff++][1]=txReplace;
			}
			
			if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][tx]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=tx;
			}
			
			if(sharedGradStrength[tyReplace=max(ty-1,0)][tx]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=tx;
			}

			if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=txReplace;
			}

			if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=min(tx+1,SOBEL_SMEM_WIDTH-1)]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=txReplace;
			}

			if(sharedGradStrength[tyReplace=min(ty+1,SOBEL_SMEM_HEIGHT-1)][txReplace=max(tx-1,0)]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=txReplace;
			}

			if(sharedGradStrength[tyReplace=max(ty-1,0)][txReplace=max(tx-1,0)]==-1){
				list[listOff][0]=tyReplace;
				list[listOff++][1]=txReplace;
			}
		}
		
	}

	tx=threadIdx.x;
	ty=threadIdx.y;
	__syncthreads();
	gradStrengthOut[ref2]=sharedGradStrength[ty+1][tx+1];
}


__device__ float reduce256( volatile float smem256[],unsigned short tID){
	__syncthreads();
	if(tID<128)	smem256[tID]+=smem256[tID+128];
	__syncthreads();
	if(tID<64)	smem256[tID]+=smem256[tID+64];
	__syncthreads();
	if(tID<32){
		smem256[tID]+=smem256[tID+32];
		smem256[tID]+=smem256[tID+16];
		smem256[tID]+=smem256[tID+8];
		smem256[tID]+=smem256[tID+4];
		smem256[tID]+=smem256[tID+2];
		smem256[tID]+=smem256[tID+1];
	}
	__syncthreads();
	return smem256[0];
}


__global__ void cannyHysteresisBlockShared(	float *gradStrength,float *gradStrengthOut,
											unsigned short iWidth,unsigned short iHeight,
											float thresholdLow, float thresholdHigh){
	
	__shared__ float sharedGradStrength[SOBEL_SMEM_HEIGHT][SOBEL_SMEM_WIDTH];

	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned int x=BLOCK_WIDTH*blockIdx.x+tx;
	unsigned int y=BLOCK_HEIGHT*blockIdx.y+ty;
	
	unsigned int ref2=iWidth*y+x;
	
	//Load center
	sharedGradStrength[ty+1][tx+1]=gradStrength[ref2];


	if(ty==0){//Load top
		sharedGradStrength[0][tx+1]=(y>0) ? (gradStrength[ref2-iWidth]): (0);
	}else if(ty==BLOCK_HEIGHT-1){//Load bottom
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][tx+1]=(y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+iWidth]): (0);
	}

	__syncthreads();

	if(tx==0){//Load left 
		sharedGradStrength[ty+1][0]=(x>0) ? (gradStrength[ref2-1]) : (0);
	}else if(tx==BLOCK_WIDTH-1){//Load right
		sharedGradStrength[ty+1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth) ? (gradStrength[ref2+1]) : (0);
	}

	__syncthreads();

	//Corners
	if(tx==0&&ty==0){
		sharedGradStrength[0][0]=(x>0&&y>0) ? (gradStrength[ref2-(iWidth+1)]) : (0);//TL
	}else if(tx==BLOCK_WIDTH-1&&ty==0){
		sharedGradStrength[0][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y>0 ) ? (gradStrength[ref2-(iWidth-1)]) : (0);//TR
	}else if(tx==0&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][0]=(x>0&&y+BLOCK_HEIGHT<iHeight) ? (gradStrength[ref2+(iWidth-1)]) : (0);//BL
	}else if(tx==BLOCK_WIDTH-1&&ty==BLOCK_HEIGHT-1){
		sharedGradStrength[SOBEL_SMEM_HEIGHT-1][SOBEL_SMEM_WIDTH-1]=(x+BLOCK_WIDTH<iWidth && y+BLOCK_HEIGHT<iHeight)? (gradStrength[ref2+(iWidth+1)]) : (0);//BR
	}


	
	__syncthreads();
	//Initialization part
	//Check if neighbors are edge pixels
	float str=sharedGradStrength[ty+1][tx+1];
	__syncthreads();

	if(str>thresholdHigh)		str=-2;		//Definite edge
	else if(str>thresholdLow)	str=-1;		//Potential edge
	else if(str>0)				str=0;		//not an edge

	++tx;
	++ty;

	__syncthreads();

	sharedGradStrength[ty][tx]=str;

	__shared__ int sharedModfied[BLOCK_HEIGHT][BLOCK_WIDTH];

	for(unsigned short a=0; a<HYST_MAX_SIZE; ++a){
		sharedModfied[ty-1][tx-1]=0;
		__syncthreads();
		//If potential edge, search neighbors for definite edge... if found, mark as definite edge 
		bool e=false;
		if(sharedGradStrength[ty][tx]==-1){
			e=e || (sharedGradStrength[ty][max(tx-1,0)]==-2);					//Left		
			e=e || (sharedGradStrength[ty][min(tx+1,SOBEL_SMEM_WIDTH-1)]==-2);	//Right
			e=e || (sharedGradStrength[max(ty-1,0)][tx]==-2);					//Top
			e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][tx]==-2);	//Bot
			e=e || (sharedGradStrength[max(ty-1,0)][max(tx-1,0)]==-2);						//Top left
			e=e || (sharedGradStrength[max(ty-1,0)][min(tx+1, SOBEL_SMEM_WIDTH-1)]==-2);	//Top right
			e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][max(tx-1,0)]==-2);					//Bot left
			e=e || (sharedGradStrength[min(ty+1,SOBEL_SMEM_HEIGHT-1)][min(tx+1, SOBEL_SMEM_WIDTH-1)]==-2);	//Bot right
		}
		__syncthreads();

		if(e){
			sharedGradStrength[ty][tx]=-2;
			sharedModfied[ty-1][tx-1]=1;
		}
		int modified=reduce256((float*)&(sharedModfied[0][0]), ty*BLOCK_WIDTH+tx);
		
		if(modified==0) break;
	}

	__syncthreads();
	gradStrengthOut[ref2]=sharedGradStrength[ty][tx];
}

__global__ void cannyBlockConverter(float *gradStrength,
									unsigned char *outputImage,
									unsigned short iWidth,unsigned short iHeight){
	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;
    unsigned int ref2=iWidth*y+x;

	//Matlab Version
	__shared__ unsigned int sharedStrength[3][BLOCK_WIDTH][BLOCK_HEIGHT];
	////Load global data into shared
	//Load center
	__syncthreads();
	float str=gradStrength[ref2];
	bool isLine=(str==-2);

	sharedStrength[0][tx][ty]=255*isLine;
	sharedStrength[1][tx][ty]=255*isLine;
	sharedStrength[2][tx][ty]=255*isLine;

	unsigned char txMod=tx-(tx/4*4);
	unsigned int globalMemAddress=(tx/4)*(iWidth*iHeight)+(iHeight*BLOCK_WIDTH*bx+BLOCK_HEIGHT*by)+iHeight*ty+txMod*4;//Every halfwarp should be multiple of 16*sizeof(type)

	x=(tx/4)*(BLOCK_WIDTH*BLOCK_HEIGHT)+txMod*4;
	__syncthreads();
	if(tx<12){
		*((unsigned int *)(outputImage+globalMemAddress))=\
			   *(sharedStrength[0][ty]+x+0)\
			+((*(sharedStrength[0][ty]+x+1))<<8)\
			+((*(sharedStrength[0][ty]+x+2))<<16)\
			+((*(sharedStrength[0][ty]+x+3))<<24);
	}
}
__global__ void cannyBlockConverter8(float *gradStrength,
									unsigned char *outputImage,
									unsigned short iWidth,unsigned short iHeight){
	unsigned char tx=threadIdx.x;
	unsigned char ty=threadIdx.y;
	unsigned char bx=blockIdx.x;
	unsigned char by=blockIdx.y;
	unsigned short int x=BLOCK_WIDTH*bx+tx;
	unsigned short int y=BLOCK_HEIGHT*by+ty;
    unsigned int ref2=iWidth*y+x;

	//Matlab Version
	__shared__ unsigned int sharedStrength[3][BLOCK_WIDTH][BLOCK_HEIGHT];
	////Load global data into shared
	//Load center
	__syncthreads();
	float str=gradStrength[ref2];
	bool isLine=(str==-2);

	sharedStrength[0][tx][ty]=255*isLine;
	sharedStrength[1][tx][ty]=255*isLine;
	sharedStrength[2][tx][ty]=255*isLine;

	unsigned int globalMemAddress=(iHeight*BLOCK_WIDTH*bx+BLOCK_HEIGHT*by)+iHeight*ty+tx*4;//Every halfwarp should be multiple of 16*sizeof(type)

	x=tx*4;
	__syncthreads();
	if(tx<4){
		*((unsigned int *)(outputImage+globalMemAddress))=\
			   *(sharedStrength[0][ty]+x+0)\
			+((*(sharedStrength[0][ty]+x+1))<<8)\
			+((*(sharedStrength[0][ty]+x+2))<<16)\
			+((*(sharedStrength[0][ty]+x+3))<<24);
	}
}


