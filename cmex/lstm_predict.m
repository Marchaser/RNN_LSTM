function yhat_out = lstm_predict(xData_t,netMexName,params,weights)
%% Clear old parameters
lstm_constant;

%% Hyper parameters
v2struct(params);

%% For prediction, batchSize=1
batchSize = 1;
batchSizeThread = 1;

%% Parameters
[xDim,sizeData] = size(xData_t);

%% Check consistent of parameters
assert(length(hDims)==nLayer);

%% Conver types
hDims = int32(hDims);

% Data
xData_t = single(xData_t);
yhat_out = zeros(yDim,sizeData,'single');

% Supply constant output data
yData = zeros(yDim,batchSize,periods,'single');
yhat_t = zeros(yDim,batchSize,periods,'single');
yhat_out = zeros(yDim,sizeData,periods,'single');

%% Initiate networks
MEX_TASK = MEX_INIT;
eval(netMexName);

%% Prediction
MEX_TASK = MEX_PREDICT;

batchStart = 1;
while (batchStart <= sizeData)
    for j=1:batchSize
        batchStartTEnd = batchStart+periods-1;
        if batchStartTEnd > sizeData
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchStartTEnd);
        batchStart = batchStart + 1;
    end
    
    if batchStartTEnd > sizeData
        break;
    end
    
    eval(netMexName);
    
    yhat_out(:,batchStart-batchSize:batchStart-1,:) = yhat_t;
end
end