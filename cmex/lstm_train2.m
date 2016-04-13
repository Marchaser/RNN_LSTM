function [weights,params] = lstm_train(xData_t,yData_t,funcLoss,dfuncLoss,params,weights)
%% Hyper parameters
% Default value
temperature = 1;
batchSize = 50;
learningRate = 0.01;
T = 10;
hDim = 100;
dropOutRate = 0.5;
numDropOuts = 1e4;
NumThreads = 4;
saveFreq = inf;
% Overwrite
if (nargin>=5)
    v2struct(params);
else
    params = v2struct(temperature,batchSize,learningRate,T,hDim,NumThreads,saveFreq);
end

% Define sigmoid function
sigmoid = @(x) 1./(1+exp(-x/temperature));

%% Data
xData_t = single(xData_t);
yData_t = single(yData_t);

%% Parameters
[xDim,Ts] = size(xData_t);
[yDim,~] = size(yData_t);

%% Initiate weights
rng(0729);

% The huge W matrix, satcking g i f o, x h together
W_gifo_x = rand_init(hDim*4,xDim);
W_gifo_h = rand_init(hDim*4,hDim);
b_gifo = rand_init(hDim*4,1);
WDim = hDim*4;

Wyh = rand_init(yDim,hDim);
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
gifo_t = zeros(hDim*4,batchSize,T,'single');
gifo_lin_t = zeros(hDim*4,batchSize,T,'single');
g_t = zeros(hDim,batchSize,T,'single');
i_t = zeros(hDim,batchSize,T,'single');
f_t = zeros(hDim,batchSize,T,'single');
o_t = zeros(hDim,batchSize,T,'single');
h_t = zeros(hDim,batchSize,T,'single');
hD_t = zeros(hDim,batchSize,T,'single');
s_t = zeros(hDim,batchSize,T,'single');
tanhs_t = zeros(hDim,batchSize,T,'single');

ylin_t = zeros(yDim,batchSize,T,'single');
yhat_t = zeros(yDim,batchSize,T,'single');
ZEROS = zeros(hDim,batchSize,'single');
ONESBatchSize = ones(batchSize,1,'single');

dh = zeros(batchSize,hDim,'single');
ds = zeros(batchSize,hDim,'single');
dyhat = zeros(batchSize,yDim,'single');
doo = zeros(batchSize,hDim,'single');
dtanhs = zeros(batchSize,hDim,'single');
dg = zeros(batchSize,hDim,'single');
di = zeros(batchSize,hDim,'single');
df = zeros(batchSize,hDim,'single');
dgifo_lin = zeros(batchSize,4*hDim,'single');
dyhat_temp = zeros(yDim,batchSize,'single');
g_temp = zeros(hDim,batchSize,'single');

dhm = zeros(batchSize,hDim,'single');
dsm = zeros(batchSize,hDim,'single');
dx_t = zeros(batchSize,xDim,T,'single');
dh_t = zeros(batchSize,hDim,T,'single');
loss_t = zeros(batchSize,T,'single');

%% Draw dropouts
% dropOutDraws_all = single(rand(hDim,numDropOuts,T) > dropOutRate);
% dropOutDraws_all = dropOutDraws_all(:)';

%% Precompute gifo_x
MEX_TRAIN=1;
MEX_TASK = MEX_TRAIN;
timeCount = tic;

xData = zeros(xDim,batchSize,T,'single');
yData = zeros(yDim,batchSize,T,'single');

batchStart = 1;

while (batchStart <= Ts)
    for j=1:batchSize
        batchEnd = batchStart+T-1;
        if batchEnd>=Ts
            break;
        end
        xData(:,j,:) = xData_t(:,batchStart:batchEnd);
        yData(:,j,:) = yData_t(:,batchStart:batchEnd);
        batchStart = batchStart + 1;
    end
    
    if batchEnd>=Ts
        break;
    end
    lstm_mex;
    
    W_gifo_x = W_gifo_x - learningRate*dW_gifo_x;
    W_gifo_h = W_gifo_h - learningRate*dW_gifo_h;
    b_gifo = b_gifo - learningRate*db_gifo;
    Wyh = Wyh - learningRate*dWyh;
    by = by - learningRate*dby;
end

weights = v2struct(W_gifo_x,W_gifo_h,b_gifo,Wyh,by);
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n,'single'))*0.1;
end
