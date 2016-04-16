function [weights,params] = lstm_train(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Clear old parameters
lstm_constant;

%% Hyper parameters
% Default value
xDim = 10;
yDim = 10;
nLayer = 1;
hDims = 10;
periods = 10;
batchSize = 64;
learningRate = 0.01;
dropoutRate = 0.5;
NumThreads = 4;
saveFreq = inf;

% RmsProp
RmsProp_gamma = 0.9;

% Overwrite
if (nargin>=4)
    v2struct(params);
else
    params = v2struct(xDim,yDim,nLayer,hDims,periods,batchSize,learningRate,dropoutRate,NumThreads);
end

%% Convert data type
hDims = int32(hDims);
learningRate = single(learningRate);
RmsProp_gamma = single(RmsProp_gamma);
xData_t = single(xData_t);
yData_t = single(yData_t);

%% Check consistent of parameters
assert(mod(batchSize,NumThreads)==0)
assert(size(xData_t,1)==xDim);
assert(size(yData_t,1)==yDim);
assert(size(xData_t,2)==size(yData_t,2));
assert(length(hDims)==nLayer);

%% Induced parameters
batchSizeThread = batchSize / NumThreads;
% Sequence data, take out the amount of periods
lengthData = size(xData_t,2) - periods+1;

%% Initiate networks
MEX_TASK = MEX_INIT;
eval(netMexName);

% sizeWeights will be returned by MEX
% Init weights
rng(0729);
weights = rand_init(sizeWeights,1);
dweights_thread = zeros([size(weights) NumThreads],'single');
RmsProp_r = ones(size(weights),'single');

% Overwrite
if (nargin>=5)
    if numel(weights_hotstart) == numel(weights)
        weights = weights_hotstart;
    else
        error('weights_hotstart size not correct');
    end
end

%% Pre treatment
MEX_TASK = MEX_PRE_TREAT;
eval(netMexName);

%% Prepare space for thread data
% Split training data based on batchSize
lengthDataBatch = floor(lengthData/batchSize);
batchDataStride = 1:lengthDataBatch:lengthDataBatch*batchSize;
% 0 based
batchDataStride = batchDataStride - 1;
% Space for loss
loss_thread = zeros(periods,batchSizeThread,NumThreads,'single');

%% Train
MEX_TASK = MEX_TRAIN;
saveCount = 0;
timeCount = tic;
for currentBatch=1:lengthDataBatch
    
    eval(netMexName);
    
    % Collapse thread dweights
    dweights = reshape(dweights_thread,[],NumThreads);
    % Sum over training set
    dweights = sum(dweights(:,1:end-1),2);
    dweights = reshape(dweights,size(weights));
    
    % Adjust learning rate
    RmsProp_r = (1-RmsProp_gamma)*dweights.^2 + RmsProp_gamma*RmsProp_r;
    RmsProp_v = learningRate ./ RmsProp_r;
    RmsProp_v = min(RmsProp_v,learningRate*5);
    RmsProp_v = max(RmsProp_v,learningRate/5);
    RmsProp_v = RmsProp_v.* dweights;
    
    %  Learn
    weights = weights - RmsProp_v;
    
    % Output
    saveCount = saveCount + 1;
    if mod(saveCount,saveFreq)==0
        output_func;
    end
    
    batchDataStride = batchDataStride + 1;
end
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n,'single'))*0.1;
end
