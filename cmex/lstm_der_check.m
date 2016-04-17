function [dweights,dweights_numerical] = lstm_der_check(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Parameters
lstm_parameters;

%% Convert Data
xData_t = cast(xData_t,typename);
yData_t = cast(yData_t,typename);

%% Induced parameters
batchSizeThread = batchSize / NumThreads;
% Sequence data, take out the amount of periods
lengthData = size(xData_t,2) - periods+1;

%% Initiate networks
MEX_TASK = MEX_NNET_INFO;
eval(netMexName);
% sizeWeights will be returned by MEX
% Init weights
rng(0729);
weights = (-1 + 2*rand(sizeWeights,1,typename))*initWeightsScale;
dweights_thread = zeros([size(weights) NumThreads],typename);
RmsProp_r = ones(size(weights),typename);

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
loss_thread = zeros(periods,batchSizeThread,NumThreads,typename);

%% Derivative check
% Evaluate at current point
MEX_TASK = MEX_INIT;
eval(netMexName);
MEX_TASK = MEX_TRAIN;
eval(netMexName);
% Collapse thread dweights
dweights = reshape(dweights_thread,[],NumThreads);
% Sum over training set
dweights = sum(dweights,2);
dweights = reshape(dweights,size(weights));
% Across periods
loss = sum(loss_thread,1);

% Evaluate at perturbation
delta = 1e-6;
dweights_numerical = zeros(1,sizeWeights,typename);
for iw = 1:sizeWeights
    % Perturb the weight a bit
    deltaWeights = delta;
    weights(iw) = weights(iw) + deltaWeights;
    
    % Evaluate Net, with fresh seed
    MEX_TASK = MEX_INIT;
    eval(netMexName);
    MEX_TASK = MEX_TRAIN;
    eval(netMexName);
    
    % Perturb back
    weights(iw) = weights(iw) - deltaWeights;
    
    % Across periods
    loss_new = sum(loss_thread,1);
    
    % Compute numerical derivative
    dweights_numerical(iw) = sum((loss_new(:) - loss(:)) / deltaWeights);
end
end
