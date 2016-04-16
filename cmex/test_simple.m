function test_simple
% A simple example
% The sequence is generated in a way that the number in sequence is the first digit of the sum
% of the previous 2 numbers
% For example
% 123583145943707741561785381909987527965167303369549

%% Generate number
xData0 = [1 2];
Ts = 1e5;
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

%% Some hyper parameters
temperature = 1;
batchSize = 64;
learningRate = 0.01;
periods = 3; % We know that only 3 periods ahead information are relevant, supply 4 to fool it
hDim = 20;
hDim1 = 20;
hDim2 = 20;
xDim = size(xData,1);
yDim = size(yData,1);
NumThreads = 4;
saveFreq = 500;
params = v2struct(temperature,batchSize,learningRate,periods,xDim,yDim,hDim,hDim1,hDim2,NumThreads,saveFreq);

%% Train
% weights = lstm_train(xData,yData,'oneLayerNet',params);
weights = lstm_train(xData,yData,'twoLayerNet',params);

%% Predict
%{
weights = load('weights');
v2struct(weights);
v2struct(weights);
weights = single([W_gifo_x(:);W_gifo_h(:);b_gifo(:);Wyh(:);by(:)]);
%}
% yhat = lstm_predict(xData(:,1:1001),'oneLayerNet',params,weights);
yhat = lstm_predict(xData(:,1:1001),'twoLayerNet',params,weights);
end