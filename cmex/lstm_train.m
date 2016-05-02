function [weights,params] = lstm_train(xData_t,yData_t,netMexName,params,weights_hotstart)
%% Parameters
clear(netMexName);
lstm_parameters;

%% Convert data
xData_t = cast(xData_t,typename);
yData_t = cast(yData_t,typename);

%% Induced parameters
batchSizeThread = batchSize / NumThreads;
% Split training set and validation set
lengthData = size(xData_t,2);
lengthTraining = floor(lengthData*trainingSet);
lengthValidation = lengthData-lengthTraining;
validationStart = lengthTraining;
% Deal with training
% Remove amount of periods
lengthTraining = lengthTraining - periods+1;
lengthTrainingBatch = floor(lengthTraining/batchSize);
assert(lengthTrainingBatch>=1);
batchTrainingStride = 1:lengthTrainingBatch:lengthTrainingBatch*batchSize;
% Zero based
batchTrainingStride = batchTrainingStride - 1;
% Deal with validation
lengthValidation = lengthValidation - periods+1;
lengthValidationBatch = floor(lengthValidation/batchSize);
assert(lengthValidationBatch>=1);
batchValidationStride0 = 1:lengthValidationBatch:lengthValidationBatch*batchSize;
% Zero based
batchValidationStride0 = validationStart + batchValidationStride0 - 1;

%% Initiate networks
% sizeWeights will be returned by MEX
% Init weights
MEX_TASK = MEX_INFO;
eval(netMexName);
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
% Space for loss
loss_thread = zeros(periods,batchSizeThread,NumThreads,typename);

%% Train level information
trainInfo.lossTraining = [];
trainInfo.lossValidation = [];

%% Train
% Count
epochCount = 0;
timeTic = tic;
last_loss_training = inf;
totalTimeTic = tic;
fprintf('Starting training, tic tic...\n');
for step=1:lengthTrainingBatch
    currentTotalSamples = step*batchSize;
    
    batchDataStride = batchTrainingStride;
    MEX_TASK = MEX_COMP_DWEIGHTS;
    eval(netMexName);
%     %{
    MEX_TASK = MEX_APPLY_WEIGHTS_CONSTR;
    eval(netMexName);
    MEX_TASK = MEX_UPDATE_WEIGHTS;
    eval(netMexName);
    %}
    
    %{
    % Apply weights decay
    dweights = dweights_thread(:,:,1) + weightsDecay*weights;
    % Call optimizer
    optim_Adam;
    %}
    
    % Output
    if mod(step,epochSize)==0
        epoch_func;
    end
    
    % Move data
    batchTrainingStride = batchTrainingStride + 1;
end
end
