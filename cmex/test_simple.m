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
xData0 = [1 2 3 4 5 6 7 8 9 0];
Ts = 3e5;
for t=1:Ts
    xNext = mod(sum(xData0(end:-1:end-5)),10);
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
periods = 6; % We know that only 3 periods ahead information are relevant, supply 4 to fool it
nLayer = 2;
% hDims = [100 100];
hDims = [100 100];
learningRate = 5e-3;
learningRateDecay = 1;
dropoutRate = 0.5;
NumThreads = 8;
MklThreads = 1;
epochSize = 500;
% typename = 'double';
typename = 'single';
nnetName = ['lstmNet_' typename];
initForgetBiases = 1;
weightsDecay = 0;
params = v2struct(xDim,yDim,nLayer,hDims,periods,batchSize,learningRate,learningRateDecay,dropoutRate,NumThreads,MklThreads,epochSize,typename,initForgetBiases,weightsDecay);

%% Derivative check
%{
clear mex;
[dweights_analytical,dweights_numerical] = lstm_der_check(xData,yData,nnetName,params);
der_err = max(abs(dweights_analytical(:) - dweights_numerical(:)))
%}

%% Train
clear mex;
weights = lstm_train(xData,yData,nnetName,params);

%% Predict
clear mex;
yhat = lstm_predict(xData(:,1:101),[],nnetName,params,weights);
end