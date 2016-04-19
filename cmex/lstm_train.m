function [weights,params] = lstm_train(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Parameters
clear(netMexName);
lstm_parameters;

%% Convert data
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

% Init optimizer
RmsProp_r = ones(size(weights),typename);
Adam_m = zeros(size(weights),typename);
Adam_v = zeros(size(weights),typename);

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
batchDataStride = 1:batchSize;
% 0 based
batchDataStride = batchDataStride - 1;
% Space for loss
loss_thread = zeros(periods,batchSizeThread,NumThreads,typename);

%% Train
% Initial warmups
dropoutRate0 = dropoutRate;
dropoutRate = 0;
hasDropoutStarted = 0;
MEX_TASK = MEX_INIT;
eval(netMexName);
% Count
saveCount = 0;
timeCount = tic;
last_loss_training = inf;
for t=1:lengthDataBatch
    currentTotalSamples = t*batchSize;
    if (currentTotalSamples>dropoutStart && hasDropoutStarted==0)
        % Start dropout
        fprintf('Starting sampling %d, starting dropout at rate %f\n',currentTotalSamples,dropoutRate0);
        dropoutRate = dropoutRate0;
        MEX_TASK = MEX_INIT;
        eval(netMexName);
        hasDropoutStarted=1;
    end
    
    MEX_TASK = MEX_TRAIN;
    eval(netMexName);
    
    % Collapse thread dweights
    dweights = reshape(dweights_thread,[],NumThreads);
    % Sum over training set
    dweights = sum(dweights(:,1:end-1),2);
    dweights = reshape(dweights,size(weights));
    
    % Optim
    % optim_RmsProp;
    optim_Adam;
    
    % Output
    saveCount = saveCount + 1;
    if mod(saveCount,saveFreq)==0
        output_func;
    end
    
    batchDataStride = batchDataStride + periods;
end
end
