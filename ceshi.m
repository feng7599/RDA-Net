
addpath('matconvnet-1.0-beta25\matlab');
addpath('SegyMAT');
%  vl_compilenn; %一次可注销
addpath(fullfile('utilities'));
tic
[output,input] = Demo_Test_FracDCNN_DAG(1,500,1,400);
t=toc;
A = output;
B = input - output;
C = input;
save('A.mat','A');
save('B.mat','B'); 
save('C.mat','C');

[output,input] = Demo_Test_FracDCNN_DAG(1,500,401,800);
A1 = output;
B1 = input - output;
C1 = input;
save('A1.mat','A1');
save('B1.mat','B1');
save('C1.mat','C1');
[output,input] = Demo_Test_FracDCNN_DAG(498,1000,1,400);
A2 = output;
B2 = input - output;
C2 = input;
save('A2.mat','A2');
save('B2.mat','B2');
save('C2.mat','C2');
[output,input] = Demo_Test_FracDCNN_DAG(498,1000,401,800);
A3 = output;
B3 = input - output;
C3 = input;
save('A3.mat','A3');
save('B3.mat','B3');
save('C3.mat','C3');

for i = 1:3
    Result   = ['A',num2str(i)];
    Residual = ['B',num2str(i)];
    load(Result)
    load(Residual)
end
A9=[A,A1;A2,A3];
B9=[B,B1;B2,B3];
C6=[C,C1;C2,C3];

load('data\pure\pure.mat')
SNR = 10*log(sum(sum(p(1:992,1:800).^2))/sum(sum((A9-p(1:992,1:800)).^2)))/log(10);




