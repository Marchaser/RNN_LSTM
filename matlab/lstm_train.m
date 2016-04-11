function [weights,params] = lstm_train(xData,yData,funcLoss,dfuncLoss,params,weights)
%% Hyper parameters
% Default value
temperature = 1;
batchSize = 50;
learningRate = 0.01;
T = 10;
gDim = 100;
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
Wgx = rand_init(gDim,xDim);
Wgh = rand_init(gDim,gDim);
bg = rand_init(gDim,1);

Wix = rand_init(gDim,xDim);
Wih = rand_init(gDim,gDim);
bi = rand_init(gDim,1);

Wfx = rand_init(gDim,xDim);
Wfh = rand_init(gDim,gDim);
bf = rand_init(gDim,1);

Wox = rand_init(gDim,xDim);
Woh = rand_init(gDim,gDim);
bo = rand_init(gDim,1);

Wyh = rand_init(yDim,gDim);
by = rand_init(yDim,1);
% Overwrite
if (nargin>=6)
    v2struct(weights);
end

%% Initiate something
countBatch = 0;
dWgx_batch = zeros(size(Wgx));
dWgh_batch = zeros(size(Wgh));
dbg_batch = zeros(size(bg));

dWix_batch = zeros(size(Wix));
dWih_batch = zeros(size(Wih));
dbi_batch = zeros(size(bi));

dWfx_batch = zeros(size(Wfx));
dWfh_batch = zeros(size(Wfh));
dbf_batch = zeros(size(bf));

dWox_batch = zeros(size(Wox));
dWoh_batch = zeros(size(Woh));
dbo_batch = zeros(size(bo));

dWyh_batch = zeros(size(Wyh));
dby_batch = zeros(size(by));

%% Loop for each sample
for tt=1:Ts
    TEnd = tt+T-1;
    if TEnd > Ts
        break;
    end
    
    % Current observation
    x_t = xData(:,tt:TEnd);
    y_t = yData(:,tt:TEnd);
    
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
    
    % Compute Loss function
    loss = sum(funcLoss(y_t,yhat_t));
    
    %% Backward propagation
    % Initiate derivative variable
    dWgx = zeros(size(Wgx));
    dWgh = zeros(size(Wgh));
    dbg = zeros(size(bg));
    
    dWix = zeros(size(Wix));
    dWih = zeros(size(Wih));
    dbi = zeros(size(bi));
    
    dWfx = zeros(size(Wfx));
    dWfh = zeros(size(Wfh));
    dbf = zeros(size(bf));
    
    dWox = zeros(size(Wox));
    dWoh = zeros(size(Woh));
    dbo = zeros(size(bo));
    
    dWyh = zeros(size(Wyh));
    dby = zeros(size(by));
    
    % Initiate dLdh, dLpds
    dhm = zeros(1,gDim);
    dsm = zeros(1,gDim);
    
    for t=T:-1:1
        % Read time series variable
        x = x_t(:,t);
        y = y_t(:,t);
        if t==1
            % Memory is truncated before time 1
            hm = zeros(gDim,1);
            sm = zeros(gDim,1);
        else
            hm = h_t(:,t-1);
            sm = s_t(:,t-1);
        end
        
        g = g_t(:,t);
        glin = glin_t(:,t);
        ii = i_t(:,t);
        ilin = ilin_t(:,t);
        f = f_t(:,t);
        flin = flin_t(:,t);
        o = o_t(:,t);
        olin = olin_t(:,t);
        
        h = h_t(:,t);
        s = s_t(:,t);
        tanhs = tanhs_t(:,t);
        
        ylin = ylin_t(:,t);
        yhat = yhat_t(:,t);
        
        backward_prop;
    end
    % derivative_check;
    
    if isnan(dWgx)
        1
    end
    
    %% Update derivative
    countBatch = countBatch+1;
    dWgx_batch = dWgx_batch+dWgx;
    dWgh_batch = dWgh_batch+dWgh;
    dbg_batch = dbg_batch+dbg;
    
    dWix_batch = dWix_batch+dWix;
    dWih_batch = dWih_batch+dWih;
    dbi_batch = dbi_batch+dbi;
    
    dWfx_batch = dWfx_batch+dWfx;
    dWfh_batch = dWfh_batch+dWfh;
    dbf_batch = dbf_batch+dbf;
    
    dWox_batch = dWox_batch+dWox;
    dWoh_batch = dWoh_batch+dWoh;
    dbo_batch = dbo_batch+dbo;
    
    dWyh_batch = dWyh_batch+dWyh;
    dby_batch = dby_batch+dby;
    if countBatch==batchSize
        Wgx = Wgx - learningRate*dWgx_batch;
        Wgh = Wgh - learningRate*dWgh_batch;
        bg = bg - learningRate*dbg_batch;
        
        Wix = Wix - learningRate*dWix_batch;
        Wih = Wih - learningRate*dWih_batch;
        bi = bi - learningRate*dbi_batch;
        
        Wfx = Wfx - learningRate*dWfx_batch;
        Wfh = Wfh - learningRate*dWfh_batch;
        bf = bf - learningRate*dbf_batch;
        
        Wox = Wox - learningRate*dWox_batch;
        Woh = Woh - learningRate*dWoh_batch;
        bo = bo - learningRate*dbo_batch;
        
        Wyh = Wyh - learningRate*dWyh_batch;
        by = by - learningRate*dby_batch;
        
        % Reset
        countBatch = 0;
        dWgx_batch = zeros(size(Wgx));
        dWgh_batch = zeros(size(Wgh));
        dbg_batch = zeros(size(bg));
        
        dWix_batch = zeros(size(Wix));
        dWih_batch = zeros(size(Wih));
        dbi_batch = zeros(size(bi));
        
        dWfx_batch = zeros(size(Wfx));
        dWfh_batch = zeros(size(Wfh));
        dbf_batch = zeros(size(bf));
        
        dWox_batch = zeros(size(Wox));
        dWoh_batch = zeros(size(Woh));
        dbo_batch = zeros(size(bo));
        
        dWyh_batch = zeros(size(Wyh));
        dby_batch = zeros(size(by));
    end
end

weights = v2struct(Wgx,Wgh,bg,Wix,Wih,bi,Wfx,Wfh,bf,Wox,Woh,bo,Wyh,by);
end

function rn = rand_init(m,n)
rn = (-1 + 2*rand(m,n))*0.1;
end
