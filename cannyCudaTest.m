function y = cannyCudaTest(fileName,thresholdLow,thresholdHigh,hysteresisIts,sigma,runs)

A = imread(fileName);

nChannels=ndims(A);
if(nChannels==2)
    nChannels=1;
   
nChannels
end    
cudaTime=0;
matlabTime=0;

for i=1:runs
    tic;
    C=canny(A,nChannels,thresholdLow,thresholdHigh,hysteresisIts,sigma);
    cudaTime=cudaTime+toc;

    tic;
    if(nChannels==3) 
        B=rgb2gray(A);
    else B=A;
    end

    D=edge(B,'canny',[thresholdLow/1024 thresholdHigh/1024],sigma);
    matlabTime=matlabTime+toc;
end

matlabTime=matlabTime/runs
cudaTime=cudaTime/runs

speedup=matlabTime/cudaTime

figure;
imshow(C);
figure;
imshow(D);

