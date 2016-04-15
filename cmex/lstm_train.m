function [weights,params] = lstm_train(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Clear old parameters
clear(netMexName);
lstm_constant;

%% Hyper parameters
% Default value
hDim = 10;
batchSize = 64;
learningRate = 0.01;
dropOutRate = 0.5;
NumThreads = 4;
saveFreq = inf;
periods = 10;

% RmsProp
RmsProp_gamma = 0.9;

% Overwrite
if (nargin>=4)
    v2struct(params);
else
    params = v2struct(batchSize,learningRate,periods,hDim,NumThreads);
end

% Convert to single
learningRate = single(learningRate);
RmsProp_gamma = single(RmsProp_gamma);

%% Data
xData_t = single(xData_t);
yData_t = single(yData_t);

%% Parameters
[xDim,lengthData] = size(xData_t);
[yDim,~] = size(yData_t);

%% Initiate weights
% Get weights size
MEX_TASK = MEX_GET_WEIGHTS_SIZE;
eval(netMexName);
rng(0729);
weights = rand_init(sizeWeights,1);
RmsProp_r = ones(size(weights),'single');

% Overwrite
if (nargin>=5)
    if numel(weights_hotstart) == numel(weights)
        weights = weights_hotstart;
    else
        error('weights_hotstart size not correct');
    end
end

%% Initiate weights
dweights_thread = zeros([size(weights) NumThreads],'single');

%% Train
% Training data
xData = zeros(xDim,batchSize,periods,'single');
yData = zeros(yDim,batchSize,periods,'single');

MEX_TASK = MEX_TRAIN;

batchStart = 1;
saveCount = 0;
timeCount = tic;

while (batchStart <= lengthData)
    for j=1:batchSize
        batchStartTEnd = batchStart+periods-1;
        if batchStartTEnd > lengthData
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchStartTEnd);
        yData(:,j,:) = yData_t(:,batchStart:batchStartTEnd);
        batchStart = batchStart + 1;
    end
    
    if batchStartTEnd > lengthData
        break;
    end
    
    eval(netMexName);
    
    % Collapse thread dweights
    dweights = sum(reshape(dweights_thread,[],NumThreads),2);
    dweights = reshape(dweights,size(weights));
    
    % Adjust learning rate
    %{
    RmsProp_r = (1-RmsProp_gamma)*dweights.^2 + RmsProp_gamma*RmsProp_r;
    RmsProp_v = learningRate ./ RmsProp_r;
    RmsProp_v = min(RmsProp_v,learningRate*5);
    RmsProp_v = max(RmsProp_v,learningRate/5);
    RmsProp_v = RmsProp_v.* dweights;
    
    weights = weights - RmsProp_v;
    %}
    weights = weights - learningRate*dweights;
    
    saveCount = saveCount + 1;
    
    if mod(saveCount,saveFreq)==0
        output_func;
    end
end
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n,'single'))*0.1;
end
