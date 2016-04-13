function [weights,params] = lstm_train(xData,yData,funcLoss,dfuncLoss,params,weights)
%% Hyper parameters
% Default value
temperature = 1;
batchSize = 50;
learningRate = 0.01;
T = 10;
gDim = 100;
dropOutRate = 0.5;
numDropOuts = 1e4;
NumThreads = 4;
saveFreq = inf;
% Overwrite
if (nargin>=5)
    v2struct(params);
else
    params = v2struct(temperature,batchSize,learningRate,T,gDim,NumThreads,saveFreq);
end

% Define sigmoid function
sigmoid = @(x) 1./(1+exp(-x/temperature));

%% Data
xData = single(xData);
yData = single(yData);

%% Parameters
[xDim,Ts] = size(xData);
[yDim,~] = size(yData);

%% Initiate weights
rng(0729);

% The huge W matrix, satcking g i f o, x h together
W_gifo_x = rand_init(gDim*4,xDim);
W_gifo_h = rand_init(gDim*4,gDim);
b_gifo = rand_init(gDim*4,1);
WDim = gDim*4;

Wyh = rand_init(yDim,gDim);
by = rand_init(yDim,1);
% Overwrite
if (nargin>=6)
    v2struct(weights);
end

%% Initiate something
dW_gifo_x = zeros(size(W_gifo_x),'single');
dW_gifo_h = zeros(size(W_gifo_h),'single');
db_gifo = zeros(size(b_gifo),'single');

dWyh = zeros(size(Wyh),'single');
dby = zeros(size(by),'single');

%% Initiate space used in computation
gifo_t = zeros(gDim*4,batchSize,T,'single');
gifo_lin_t = zeros(gDim*4,batchSize,T,'single');
g_t = zeros(gDim,batchSize,T,'single');
i_t = zeros(gDim,batchSize,T,'single');
f_t = zeros(gDim,batchSize,T,'single');
o_t = zeros(gDim,batchSize,T,'single');
h_t = zeros(gDim,batchSize,T,'single');
hD_t = zeros(gDim,batchSize,T,'single');
s_t = zeros(gDim,batchSize,T,'single');
tanhs_t = zeros(gDim,batchSize,T,'single');

ylin_t = zeros(yDim,batchSize,T,'single');
yhat_t = zeros(yDim,batchSize,T,'single');
ZEROS = zeros(gDim,batchSize,'single');
ONESBatchSize = ones(batchSize,1,'single');

dh = zeros(batchSize,gDim,'single');
dhD = zeros(batchSize,gDim,'single');
ds = zeros(batchSize,gDim,'single');
dyhat = zeros(batchSize,yDim,'single');
doo = zeros(batchSize,gDim,'single');
dtanhs = zeros(batchSize,gDim,'single');
dg = zeros(batchSize,gDim,'single');
di = zeros(batchSize,gDim,'single');
df = zeros(batchSize,gDim,'single');
dgifo_lin = zeros(batchSize,4*gDim,'single');
dyhat_temp = zeros(yDim,batchSize,'single');
g_temp = zeros(gDim,batchSize,'single');

%% Draw dropouts
dropOutDraws_all = single(rand(gDim,numDropOuts,T) > dropOutRate);
dropOutDraws_all = dropOutDraws_all(:)';

%% Precompute gifo_x
gifo_x_t = zeros(gDim*4,T+batchSize-1,'single');

MEX_TRAIN=1;
MEX_TASK = MEX_TRAIN;
timeCount = tic;

lstm_mex;

weights = v2struct(W_gifo_x,W_gifo_h,b_gifo,Wyh,by);
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n,'single'))*0.1;
end
