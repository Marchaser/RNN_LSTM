function yhat_out = lstm_predict(xData,params,weights)
%% Hyper parameters
v2struct(params);
% Define sigmoid function
sigmoid = @(x) 1./(1+exp(-x/temperature));

%% Parameters
[xDim,Ts] = size(xData);

%% Weights
v2struct(weights);
[yDim,~] = size(Wyh);
yhat_out = zeros(yDim,Ts);

for tt=1:Ts
    TEnd = tt+T-1;
    if TEnd > Ts
        break;
    end
    
    % Current observation
    x_t = xData(:,tt:TEnd);
    
    % Prepare time series for computation
    g_t = zeros(gDim,T);
    glin_t = zeros(gDim,T);
    i_t = zeros(gDim,T);
    ilin_t = zeros(gDim,T);
    f_t = zeros(gDim,T);
    flin_t = zeros(gDim,T);
    o_t = zeros(gDim,T);
    olin_t = zeros(gDim,T);
    
    h_t = zeros(gDim,T);
    s_t = zeros(gDim,T);
    tanhs_t = zeros(gDim,T);
    
    ylin_t = zeros(yDim,T);
    yhat_t = zeros(yDim,T);

    %% Forward pass
    for t=1:T
        % Read time series variable
        x = x_t(:,t);
        if t==1
            % Memory is truncated before time 1
            hm = zeros(gDim,1);
            sm = zeros(gDim,1);
        else
            hm = h_t(:,t-1);
            sm = s_t(:,t-1);
        end
        
        forward_pass;
        
        % Write to time series variable
        g_t(:,t) = g;
        glin_t(:,t) = glin;
        i_t(:,t) = ii;
        ilin_t(:,t) = ilin;
        f_t(:,t) = f;
        flin_t(:,t) = flin;
        o_t(:,t) = o;
        olin_t(:,t) = olin;
        
        h_t(:,t) = h;
        s_t(:,t) = s;
        tanhs_t(:,t) = tanhs;
        
        ylin_t(:,t) = ylin;
        yhat_t(:,t) = yhat;
    end
    
    % Write to predicted variable
    yhat_out(:,TEnd) = yhat_t(:,end);
end
end