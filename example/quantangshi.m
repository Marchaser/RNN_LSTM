function quantangshi
if isunix
    run '../SET_PATH.m';
end

text = fileread('quantangshi.txt');
text = text(1:1e6);
fprintf('Text length: %d\n', length(text));
[text_dic,~,text_code] = unique(text);
text_linear_code = dummyvar(text_code)';

%% Train the network
xDim = size(text_linear_code,1);
yDim = size(text_linear_code,1);
nLayer = 2;
hDims = [512 512];
batchSize = 64;
periods = 100;
learningRate = 5e-3;
learningRateDecay = 1;
dropoutRate = 0.5;
dropoutStart = 0;
weightsDecay = 1e-3*batchSize;

trainingSet = 0.999;
validationSet = 1-trainingSet;

typename = 'single';
nnetName = ['lstmNet_' typename];
clear(nnetName);

epochSize = 200;

NumThreads = 4;
NSlots = getenv('NSLOTS');
if (strcmp(NSlots, '') == 0)
    NumThreads = str2double(NSlots);
    fprintf('NumThreads: %d\n',NumThreads);
end

params = v2struct(xDim,yDim,batchSize,periods,nLayer,hDims,NumThreads,learningRate,dropoutRate, ...
    dropoutStart,NumThreads,epochSize,learningRateDecay,typename,trainingSet,validationSet,weightsDecay);

xData = text_linear_code(:,1:end-1);
yData = text_linear_code(:,2:end);


%% Train
addpath('../cmex');
weights = lstm_train(xData,yData,nnetName,params);
end