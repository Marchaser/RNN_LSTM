function [weights,params] = lstm_train(xData,yData,funcLoss,dfuncLoss,params,weights)
%% Hyper parameters
% Default value
temperature = 1;
batchSize = 50;
learningRate = 0.01;
T = 10;
gDim = 100;
dropOutRate = 0.5;
% Overwrite
if (nargin>=5)
    v2struct(params);
else
    params = v2struct(temperature,batchSize,learningRate,T,gDim);
end

% Define sigmoid function
sigmoid = @(x) 1./(1+exp(-x/temperature));

%% Parameters
[xDim,Ts] = size(xData);
[yDim,~] = size(yData);

%% Initiate weights
rng(0729);

% The huge W matrix, satcking g i f o, x h together
W_gifo_x = rand_init(gDim*4,xDim);
W_gifo_h = rand_init(gDim*4,gDim);
b_gifo = rand_init(gDim*4,1);

Wyh = rand_init(yDim,gDim);
by = rand_init(yDim,1);
% Overwrite
if (nargin>=6)
    v2struct(weights);
end

%% Initiate something
dW_gifo_x = zeros(size(W_gifo_x));
dW_gifo_h = zeros(size(W_gifo_h));
db_gifo = zeros(size(b_gifo));

dWyh = zeros(size(Wyh));
dby = zeros(size(by));

%% Initiate space used in computation
gifo_t = zeros(gDim*4,batchSize,T);
gifo_lin_t = zeros(gDim*4,batchSize,T);
h_t = zeros(gDim,batchSize,T);
s_t = zeros(gDim,batchSize,T);
tanhs_t = zeros(gDim,batchSize,T);

ylin_t = zeros(yDim,batchSize,T);
yhat_t = zeros(yDim,batchSize,T);

%% Precompute gifo_x
gifo_x = zeros(gDim*4,T+batchSize-1);
gifo_x(:,end-T+2:end) = W_gifo_x * xData(:,1:T-1);

%% Loop for each batch of sample
batchStart = 1;
pNewDataStart = T;
while batchStart<=Ts
    batchEnd = batchStart + batchSize-1;
    pDataEnd = batchEnd+T-1;
    if pDataEnd>Ts
        break;
    end
    
    %% Compute gifo_x
    gifo_x(:,1:T-1) = gifo_x(:,end-T+2:end);
    gifo_x(:,T:end) = W_gifo_x * xData(:,pNewDataStart:pDataEnd);
    
    %% Extract x, y
    x = xData(:,batchStart:pDataEnd);
    y = yData(:,batchStart:pDataEnd);
    
    %% Shift pointer
    batchStart = batchEnd+1;
    pNewDataStart = pDataEnd+1;
    
    % Prepare time series for computation
    g_t = zeros(gDim,batchSize,T);
    glin_t = zeros(gDim,batchSize,T);
    i_t = zeros(gDim,batchSize,T);
    ilin_t = zeros(gDim,batchSize,T);
    f_t = zeros(gDim,batchSize,T);
    flin_t = zeros(gDim,batchSize,T);
    o_t = zeros(gDim,batchSize,T);
    olin_t = zeros(gDim,batchSize,T);
    
    h_t = zeros(gDim,batchSize,T);
    s_t = zeros(gDim,batchSize,T);
    tanhs_t = zeros(gDim,batchSize,T);
    
    ylin_t = zeros(yDim,batchSize,T);
    yhat_t = zeros(yDim,batchSize,T);
   
    %% Forward pass
    for t=1:T
        if t==1
            % Memory is truncated before time 1
            hm = zeros(gDim,batchSize,1);
            sm = zeros(gDim,batchSize,1);
        else
            hm = h_t(:,:,t-1);
            sm = s_t(:,:,t-1);
        end
        
        forward_pass;
        
        % Write to time series variable
        g_t(:,:,t) = g;
        glin_t(:,:,t) = glin;
        i_t(:,:,t) = ii;
        ilin_t(:,:,t) = ilin;
        f_t(:,:,t) = f;
        flin_t(:,:,t) = flin;
        o_t(:,:,t) = o;
        olin_t(:,:,t) = olin;
        
        h_t(:,:,t) = h;
        s_t(:,:,t) = s;
        tanhs_t(:,:,t) = tanhs;
        
        ylin_t(:,:,t) = ylin;
        yhat_t(:,:,t) = yhat;
    end
    
    %% Backward propagation
    % Initiate derivative variable
    dW_gifo_x = zeros(size(W_gifo_x));
    dW_gifo_h = zeros(size(W_gifo_h));
    db_gifo = zeros(size(b_gifo));
    
    dWyh = zeros(size(Wyh));
    dby = zeros(size(by));
    
    % Initiate dLdh, dLpds
    dhm = zeros(batchSize,gDim);
    dsm = zeros(batchSize,gDim);
    
    for t=T:-1:1
        if t==1
            % Memory is truncated before time 1
            hm = zeros(gDim,batchSize,1);
            sm = zeros(gDim,batchSize,1);
        else
            hm = h_t(:,:,t-1);
            sm = s_t(:,:,t-1);
        end
        
        g = g_t(:,:,t);
        glin = glin_t(:,:,t);
        ii = i_t(:,:,t);
        ilin = ilin_t(:,:,t);
        f = f_t(:,:,t);
        flin = flin_t(:,:,t);
        o = o_t(:,:,t);
        olin = olin_t(:,:,t);
        
        h = h_t(:,:,t);
        s = s_t(:,:,t);
        tanhs = tanhs_t(:,:,t);
        
        ylin = ylin_t(:,:,t);
        yhat = yhat_t(:,:,t);
        
        backward_prop;
    end
    
    %% Update derivative
    W_gifo_x = W_gifo_x - learningRate*dW_gifo_x;
    W_gifo_h = W_gifo_h - learningRate*dW_gifo_h;
    b_gifo = b_gifo - learningRate*db_gifo;
    
    Wyh = Wyh - learningRate*dWyh;
    by = by - learningRate*dby;
end

weights = v2struct(W_gifo_x,W_gifo_h,b_gifo,Wyh,by);
end

function rn = rand_init(m,n)
rn = double((-1 + 2*rand(m,n,'single'))*0.1);
end
