% Hyper parameters
%% Clear old parameters
lstm_constant;

%% Hyper parameters
% Data type
typename = 'double';

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
MklThreads = 1;
epochSize = inf;
dropoutSeed = 823;

temperature =  1;

weightsDecay = 0;

% Data set
trainingSet = 0.95;
validationSet = 0.05;

% Init parameters scale
initWeightsScale = 0.1;
initForgetBiases = 0;

learningRateDecay = 0.95;

% dropout starting periods
dropoutStart = 0;

% RmsProp
RmsPropDecay = 0.9;

% Adam
Adam_beta1 = 0.9;
Adam_beta2 = 0.999;
Adam_epsilon=1e-8;

% Overwrite
if (nargin>=4)
    allowedParamsName = who;
    paramsName = fieldnames(params);
    isAllowedParamsName = ismember(paramsName,allowedParamsName);
    whichParamsNameNotAllowed = find(isAllowedParamsName==0);
    if ~isempty(whichParamsNameNotAllowed)
        error(['Params not allowed: ' paramsName{whichParamsNameNotAllowed(1)}]);
    end
    v2struct(params);
else
    params = v2struct(xDim,yDim,nLayer,hDims,periods,batchSize,learningRate,dropoutRate,dropoutSeed,NumThreads);
end

%% Convert data type
hDims = int32(hDims);

%% Check consistent of parameters
assert(mod(batchSize,NumThreads)==0)
assert(size(xData_t,1)==xDim);
assert(size(yData_t,1)==yDim);
assert(size(xData_t,2)==size(yData_t,2));
assert(length(hDims)==nLayer);