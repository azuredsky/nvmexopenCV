%by sky.chen
%generate canny.mexw
nvmex('canny.cu')
%cannyCudaTest(fileName,thresholdLow,thresholdHigh,hysteresisIts,sigma,runs)
%help edge
cannyCudaTest('3.jpg',.05*1024,.1*1024,32,sqrt(2),10)