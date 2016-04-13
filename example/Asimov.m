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
batchSize = 100;
learningRate = 0.00005;
T = 100;
gDim = 256;

NumThreads = 8;
%{
NSlots = getenv('NSLOTS');
if (strcmp(NSlots, '') == 0)
    NumThreads = str2double(NSlots)
end
%}

saveFreq = 100;
params = v2struct(temperature,batchSize,learningRate,T,gDim,NumThreads,saveFreq);

xData = text_linear_code(:,1:end-1);
yData = text_linear_code(:,2:end);

%% Loss function
funcLoss = @(y,yhat) -sum( y.*log(yhat), 1 );
dfuncLoss = @(y,yhat) [yhat - y]';

%% Train
addpath('../cmex');
weights = lstm_train(xData,yData,funcLoss,dfuncLoss,params);
end