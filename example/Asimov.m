function Asimov
if isunix
    run '../SET_PATH.m';
end

text = fileread('Foundation - Isaac Asimov.txt');
% text = text(1:1e6);
fprintf('Text length: %d\n', length(text));
[text_dic,~,text_code] = unique(text);
text_linear_code = dummyvar(text_code)';

%% Train the network
xDim = size(text_linear_code,1);
yDim = size(text_linear_code,1);
nLayer = 2;
hDims = [256 256];
batchSize = 64;
periods = 100;
learningRate = 5e-4;
dropOutRate = 0.5;

saveFreq = 100;

% NumThreads = 8;
NSlots = getenv('NSLOTS');
if (strcmp(NSlots, '') == 0)
    NumThreads = str2double(NSlots);
end
fprintf('NumThreads: %d\n',NumThreads);

params = v2struct(xDim,yDim,batchSize,periods,nLayer,hDims,NumThreads,learningRate,dropOutRate,NumThreads,saveFreq);

xData = text_linear_code(:,1:end-1);
yData = text_linear_code(:,2:end);


%% Train
addpath('../cmex');
weights = lstm_train(xData,yData,'lstmNet',params);
end