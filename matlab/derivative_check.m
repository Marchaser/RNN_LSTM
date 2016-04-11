% derivative_check
der_delta = 1e-6;

for t=1:T
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
loss1 = sum(funcLoss(y_t,yhat_t));

Wyh(1) = Wyh(1) + der_delta;
for t=1:T
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
loss2 = sum(funcLoss(y_t,yhat_t));

der = (loss2 - loss1) / der_delta;
% Wgx(1) = Wgx(1) - der_delta;