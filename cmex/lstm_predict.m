function yhat_out = lstm_predict(xData_t,yData_t,netMexName,params,weights)
%% Parameters
if isempty(yData_t)
    yData_t = zeros(size(xData_t),class(xData_t));
end

%% Clear old parameters
lstm_parameters;

%% For prediction, batchSize=1
batchSize = 1;
batchSizeThread = 1;

%% Data
xData_t = cast(xData_t,typename);
lengthData = size(xData_t,2);
yhat_out = zeros(yDim,lengthData,typename);

% Supply constant output data
yData = zeros(yDim,batchSize,periods,typename);
yhat_t = zeros(yDim,batchSize,periods,typename);
yhat_out = zeros(yDim,lengthData,periods,typename);

%% Initiate networks
MEX_TASK = MEX_INFO;
eval(netMexName);
% Make sure sizeWeights is correct
assert(numel(weights)==sizeWeights);

%% Prediction
MEX_TASK = MEX_PREDICT;

batchStart = 1;
while (batchStart <= lengthData)
    for j=1:batchSize
        batchStartTEnd = batchStart+periods-1;
        if batchStartTEnd > lengthData
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchStartTEnd);
        batchStart = batchStart + 1;
    end
    
    if batchStartTEnd > lengthData
        break;
    end
    
    eval(netMexName);
    
    yhat_out(:,(batchStart-batchSize):(batchStart-1),:) = yhat_t;
end
end