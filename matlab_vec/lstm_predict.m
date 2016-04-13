function yhat_out = lstm_predict(xData,params,weights)
%% Hyper parameters
v2struct(params);
% Define sigmoid function
sigmoid = @(x) 1./(1+exp(-x/temperature));
% Predict one by one
batchSize = 1;

%% Parameters
[xDim,Ts] = size(xData);

%% Weights
v2struct(weights);
[yDim,~] = size(Wyh);
yhat_out = zeros(yDim,Ts);

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
        
        dropOutDraws = ones(gDim,batchSize);
        
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
    
    % Write to predicted variable
    yhat_out(:,pDataEnd) = yhat_t(:,end);
end
end