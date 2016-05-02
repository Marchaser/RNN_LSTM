function [dweights_analytical,dweights_numerical] = lstm_der_check(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Parameters
clear(netMexName);
lstm_parameters;

%% Convert Data
xData_t = cast(xData_t,typename);
yData_t = cast(yData_t,typename);

%% Induced parameters
batchSizeThread = batchSize / NumThreads;
% Sequence data, take out the amount of periods
lengthData = size(xData_t,2) - periods+1;

%% Initiate networks
MEX_TASK = MEX_INFO;
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
batchDataStride0 = 1:lengthDataBatch:lengthDataBatch*batchSize;
% 0 based
batchDataStride0 = batchDataStride0 - 1;
% Space for loss
loss_thread = zeros(periods,batchSizeThread,NumThreads,typename);

%% Derivative check
checkSize = 3;

% Eavluate at Current points
dweights_analytical = zeros(sizeWeights,checkSize,typename);
loss = zeros(1,batchSizeThread,NumThreads,checkSize,typename);
% Evaluate Net, with fresh seed and memory
MEX_TASK = MEX_INIT_SEED;
eval(netMexName);
MEX_TASK = MEX_INIT_MEMORY;
eval(netMexName);

batchDataStride = batchDataStride0;
for j=1:checkSize
    MEX_TASK = MEX_INIT_SEED;
    eval(netMexName);
    MEX_TASK = MEX_INIT_MEMORY;
    eval(netMexName);
    MEX_TASK = MEX_COMP_DWEIGHTS;
    eval(netMexName);

    % Store dweights
    dweights_analytical(:,j) = dweights_thread(:,:,1);
    % Across periods
    loss(:,:,:,j) = sum(loss_thread,1);
    
    batchDataStride = batchDataStride + 1;
end

% Evaluate at perturbation
delta = 1e-6;
dweights_numerical = zeros(sizeWeights,checkSize,typename);
for iw = 1:sizeWeights
    % Perturb the weight a bit
    deltaWeights = delta;
    weights(iw) = weights(iw) + deltaWeights;
    
    batchDataStride = batchDataStride0;
    for j=1:checkSize
        MEX_TASK = MEX_INIT_SEED;
        eval(netMexName);
        MEX_TASK = MEX_INIT_MEMORY;
        eval(netMexName);
        MEX_TASK = MEX_COMP_DWEIGHTS;
        eval(netMexName);
        
        % Across periods
        loss_new = sum(loss_thread,1);
        
        % Compute numerical derivative
        dweights_numerical_thread = sum(reshape(loss_new,[],NumThreads)-reshape(loss(:,:,:,j),[],NumThreads),1) / deltaWeights;
        dweights_numerical(iw,j) = sum(dweights_numerical_thread,2);
        
        batchDataStride = batchDataStride + 1;
    end
    % Perturb back
    weights(iw) = weights(iw) - deltaWeights;
end
end
