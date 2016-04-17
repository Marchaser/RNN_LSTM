function test_simple
% A simple example
% The sequence is generated in a way that the number in sequence is the first digit of the sum
% of the previous 2 numbers
% For example
% 123583145943707741561785381909987527965167303369549

%% Generate number
%{
xData0 = [1 2 3 4 5 6 7 8 9 0];
Ts = 2e5;
for t=1:Ts
    xNext = mod(sum(xData0(end:-1:end-9)),10);
    xData0 = [xData0 xNext];
end
%}
xData0 = [1 2];
Ts = 2e5;
for t=1:Ts
    xNext = mod(sum(xData0(end:-1:end-1)),10);
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

%% Some hyper parameters
xDim = size(xData,1);
yDim = size(yData,1);
batchSize = 64;
periods = 10; % We know that only 3 periods ahead information are relevant, supply 4 to fool it
nLayer = 2;
hDims = [100 100];
learningRate = 0.1;
dropoutRate = 0.5;
NumThreads = 4;
saveFreq = 500;
params = v2struct(xDim,yDim,nLayer,hDims,periods,batchSize,learningRate,dropoutRate,NumThreads,saveFreq);

%% Derivative check
% clear lstmNet;
% lstm_der_check(xData,yData,'lstmNet',params);

%% Train
clear lstmNet;
weights = lstm_train(xData,yData,'lstmNet',params);

%% Predict
yhat = lstm_predict(xData(:,1:1001),[],'lstmNet',params,weights);
end