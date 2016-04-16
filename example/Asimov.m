function Asimov
if isunix
    run '../SET_PATH.m';
end

text = fileread('t8.shakespeare.txt');
text = text(1:1e6);
[text_dic,~,text_code] = unique(text);
text_linear_code = dummyvar(text_code)';

%% Train the network
temperature = 1;
batchSize = 64;
learningRate = 5e-4;
periods = 100;
hDim = 256;
hDim1 = 256;
hDim2 = 256;

NumThreads = 8;
%{
NSlots = getenv('NSLOTS');
if (strcmp(NSlots, '') == 0)
    NumThreads = str2double(NSlots)
end
%}

saveFreq = 100;
params = v2struct(temperature,batchSize,learningRate,periods,hDim,hDim1,hDim2,NumThreads,saveFreq);

xData = text_linear_code(:,1:end-1);
yData = text_linear_code(:,2:end);


%% Train
addpath('../cmex');
% weights = lstm_train(xData,yData,'oneLayerNet',params);
weights = lstm_train(xData,yData,'twoLayerNet',params);
end