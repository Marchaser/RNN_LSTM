function test_simple
% A simple example
% The sequence is generated in a way that the number in sequence is the first digit of the sum
% of the previous 2 numbers
% For example
% 123583145943707741561785381909987527965167303369549

%% Generate number
xData0 = [1 2];
Ts = 1e4;
for t=1:Ts
    xNext = mod(xData0(end)+xData0(end-1),10);
    xData0 = [xData0 xNext];
end

% Convert number to 1 based
xData = xData0+1;

% Convert number to linear independent
I = eye(max(xData));
xData = I(:,xData);

% yData is just the next number, make sure make the same length
yData = xData(:,2:end);
xData = xData(:,1:end-1);

%% Loss function
funcLoss = @(y,yhat) -sum( y.*log(yhat), 1 );
dfuncLoss = @(y,yhat) [yhat - y]';

%% Some hyper parameters
temperature = 1;
batchSize = 64;
learningRate = 0.01;
T = 100; % We know that only 3 periods ahead information are relevant, supply 4 to fool it
hDim = 128;
xDim = size(xData,1);
yDim = size(yData,1);
NumThreads = 4;
saveFreq = 50;
params = v2struct(temperature,batchSize,learningRate,T,xDim,yDim,hDim,NumThreads,saveFreq);

%% Train
weights = lstm_train(xData,yData,funcLoss,dfuncLoss,params);

%% Predict
yhat = lstm_predict(xData(:,1:1001),params,weights);

yhat(:,:,end)
end
