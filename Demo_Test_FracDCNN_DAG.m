function [output,input] = Demo_Test_FracDCNN_DAG(row,row1,column,column1)
folder_test         = 'data\test';

showresult  = 1;
gpu         = 1;

load(fullfile(['model\model.mat']));

net = dagnn.DagNN.loadobj(net) ;

net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;

net.mode = 'test';

if gpu
    net.move('gpu');
end

ext         =  {'*.mat'}; 
filePaths_test   =  [];

for i = 1 : length(ext)
    filePaths_test = cat(1,filePaths_test, dir(fullfile(folder_test, ext{i})));
end


SNRs = zeros(1,length(filePaths_test));
SSIMs = zeros(1,length(filePaths_test));

for i = 1 : length(filePaths_test)
    
    label = load(fullfile(folder_test,filePaths_test(i).name)); 
    label = struct2cell(label);
    label = cell2mat(label);
   
    label = label(row:row1,column:column1); 

    label = modcrop(label,8);
    input = single(label);
    if gpu
        gpu_input = gpuArray(input);
    end
    tic
    net.eval({'input', gpu_input}) ;
    toc
    output = gather(squeeze(gather(net.vars(out1).value)));

    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,output,0,0);

    if showresult
                       
       pause(1)
    end
   SNRs(i) = PSNRCur;disp('峰值信噪比');
   disp([num2str(PSNRCur,'%2.2f'),'dB']);
   SSIMs(i) = SSIMCur;disp('结构相似度');
   disp([num2str(SSIMCur,'%2.4f')]);
disp('测试已完成')
end





