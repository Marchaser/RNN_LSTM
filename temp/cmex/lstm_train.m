function [weights,params] = lstm_train(xData_t,yData_t,funcLoss,dfuncLoss,params,weights_hotstart)
%% Clear old parameters
clear mex;

%% Hyper parameters
% Default value
temperature = 1;
batchSize = 50;
learningRate = 0.01;
T = 10;
hDim = 100;
dropOutRate = 0.5;
NumThreads = 4;
saveFreq = inf;

% RmsProp
RmsProp_gamma = 0.9;

% Convert to single
learningRate = single(learningRate);
RmsProp_gamma = single(RmsProp_gamma);

% Overwrite
if (nargin>=5)
    v2struct(params);
else
    params = v2struct(temperature,batchSize,learningRate,T,hDim,NumThreads,saveFreq);
end

%% Data
xData_t = single(xData_t);
yData_t = single(yData_t);

%% Parameters
[xDim,Ts] = size(xData_t);
[yDim,~] = size(yData_t);

%% Initiate weights
rng(0729);

% The huge W matrix, satcking g i f o, x h together
W_gifo_x = rand_init(hDim*4,xDim);
W_gifo_h = rand_init(hDim*4,hDim);
b_gifo = rand_init(hDim*4,1);
Wyh = rand_init(yDim,hDim);
by = rand_init(yDim,1);
% Stack all weights together
weights = [W_gifo_x(:);W_gifo_h(:);b_gifo(:);Wyh(:);by(:)];
numWeights = numel(weights);
RmsProp_r = ones(numWeights,1);

% Overwrite
if (nargin>=6)
    v2struct(weights_hotstart);
end

%% Initiate weights
dweights_thread = zeros([size(weights) NumThreads],'single');

%% Preallocate space
MEX_TRAIN=1;
MEX_COMPUTE_MEMORY_SIZE = 3;
% MEX_TASK = MEX_COMPUTE_MEMORY_SIZE;
% batchSize = batchSize/NumThreads;
% lstm_mex;
% batchSize = batchSize*NumThreads;
% memory_thread = zeros(memorySize,NumThreads,'single');

MEX_TASK = MEX_TRAIN;
timeCount = tic;

xData = zeros(xDim,batchSize,T,'single');
yData = zeros(yDim,batchSize,T,'single');

batchStart = 1;
saveCount = 0;

while (batchStart <= Ts)
    for j=1:batchSize
        batchStartTEnd = batchStart+T-1;
        if batchStartTEnd > Ts
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchStartTEnd);
        yData(:,j,:) = yData_t(:,batchStart:batchStartTEnd);
        batchStart = batchStart + 1;
    end
    
    if batchStartTEnd > Ts
        break;
    end
    lstm_mex;
    
    % Collapse thread dweights
    dweights = sum(reshape(dweights_thread,[],NumThreads),2);
    dweights = reshape(dweights,size(weights));
    
    % Adjust learning rate
    RmsProp_r = (1-RmsProp_gamma)*dweights.^2 + RmsProp_gamma*RmsProp_r;
    RmsProp_v = learningRate ./ RmsProp_r;
    RmsProp_v = max(RmsProp_v,learningRate*5);
    RmsProp_v = min(RmsProp_v,learningRate/5);
    RmsProp_v = RmsProp_v.* dweights;
    
    weights = weights - RmsProp_v;
    
    saveCount = saveCount + 1;
    
    if mod(saveCount,saveFreq)==0
        output_func;
    end
end
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n,'single'))*0.1;
end
