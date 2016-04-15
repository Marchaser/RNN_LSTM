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
learningRate = 0.1;
T = 3; % We know that only 3 periods ahead information are relevant, supply 4 to fool it
gDim = 20;
params = v2struct(temperature,batchSize,learningRate,T,gDim);

%% Train
weights = lstm_train(xData,yData,funcLoss,dfuncLoss,params);

%% Load
%{
weights_other = load('weights_150_9600');
weights_other = weights_other.weights;
xDim = 10;
yDim = 10;
hDim = gDim;
W_gifo_x = weights_other(1:xDim*gDim*4);
W_gifo_h = weights_other(xDim*gDim*4+1:xDim*gDim*4+gDim*gDim*4);
b_gifo = weights_other(xDim*gDim*4+gDim*gDim*4+1:xDim*gDim*4+gDim*gDim*4+gDim*4);
Wyh = weights_other(xDim*gDim*4+gDim*gDim*4+gDim*4+1:xDim*gDim*4+gDim*gDim*4+gDim*4+yDim*hDim);
by = weights_other(xDim*gDim*4+gDim*gDim*4+gDim*4+yDim*hDim+1:xDim*gDim*4+gDim*gDim*4+gDim*4+yDim*hDim+yDim);
W_gifo_x = reshape(W_gifo_x,gDim*4,xDim);
W_gifo_h = reshape(W_gifo_h,gDim*4,gDim);
b_gifo = reshape(b_gifo,gDim*4,1);
Wyh = reshape(Wyh,yDim,hDim);
by = reshape(by,yDim,1);
weights = v2struct(W_gifo_x,W_gifo_h,b_gifo,Wyh,by);
%}


%% Predict
yhat = lstm_predict(xData(:,1:21),params,weights);
end