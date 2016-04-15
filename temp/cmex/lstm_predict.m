function yhat_out = lstm_predict(xData_t,params,weights)
%% Hyper parameters
v2struct(params);

%% For prediction, batchSize=1
batchSize = 1;

%% Parameters
[xDim,Ts] = size(xData_t);

% Data
xData_t = single(xData_t);
yhat_out = zeros(yDim,Ts,'single');

% Supply constant output data
yData = zeros(yDim,batchSize,T,'single');
yhat_t = zeros(yDim,batchSize,T,'single');
yhat_out = zeros(yDim,Ts,T,'single');

MEX_TRAIN = 1;
MEX_PREDICT = 2;
MEX_COMPUTE_MEMORY_SIZE = 3;
MEX_TASK = MEX_PREDICT;

batchStart = 1;
while (batchStart <= Ts)
    for j=1:batchSize
        batchStartTEnd = batchStart+T-1;
        if batchStartTEnd > Ts
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchStartTEnd);
        batchStart = batchStart + 1;
    end
    
    if batchStartTEnd > Ts
        break;
    end
    lstm_mex;
    
    yhat_out(:,batchStart-batchSize:batchStart-1,:) = yhat_t;
end
end