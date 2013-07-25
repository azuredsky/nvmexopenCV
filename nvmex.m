function nvmex(cuFileName)
%NVMEX Compiles and links a CUDA file for MATLAB usage
%   NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
%   invoking the CUDA compiler, nvcc, and then linking with the MEX
%   function in MATLAB.

% Copyright 2009 The MathWorks, Inc.

% !!! Modify the paths below to fit your own installation !!!
if ispc % Windows
    CUDA_LIB_Location = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib\Win32';
    gpu_Opencv_LIB = 'E:\opencv\GPU\lib\Release';
    Host_Compiler_Location = '-ccbin "D:\Program Files\Microsoft Visual Studio 10.0\VC\bin"';
    PIC_Option = '';
    aa=' -I"D:\Program Files\MATLAB\R2013a\extern\include"';
    inc=' -I"E:\opencv\build\include"';
else % Mac and Linux (assuming gcc is on the path)
    CUDA_LIB_Location = '/usr/local/cuda/lib64';
    Host_Compiler_Location = '';
    PIC_Option = ' --compiler-options -fPIC ';
end
% !!! End of things to modify !!!

[path,filename,zaet] = fileparts(cuFileName);

nvccCommandLine = [ ...
    'nvcc --compile ' cuFileName ' ' Host_Compiler_Location ' ' ...
    ' -o ' filename '.o ' ...
    PIC_Option ...
    inc...
    aa ...
    ];

%mexCommandLine = ['mex (''' filename '.o'', ''-L' CULA_LIB ''', ''-L' CUDA_LIB_Location ''', ''-lcudart'',''-lcufft'', ''-lcula_core'',''-lcula_lapack'',''-lcublas'')'];
mexCommandLine = ['mex (''' filename '.o'', ''-L' gpu_Opencv_LIB ''', ''-L' CUDA_LIB_Location ''', ''-lcudart'',''-lcufft'',''-lopencv_core245'', ''-lopencv_gpu245'',''-lopencv_highgui245'',''-lopencv_imgproc245'')'];
disp(nvccCommandLine);
status = system(nvccCommandLine);
if status < 0
    error 'Error invoking nvcc';
end

disp(mexCommandLine);
eval(mexCommandLine);

end
