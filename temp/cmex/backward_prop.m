% Backward propagation for one time step
% Back through time
dyhat = dfuncLoss(y(:,t:t+batchSize-1),yhat);
dWyh = dWyh + dyhat'*h';
dby = dby + sum(dyhat,1)';
dLdh = dhm + dyhat * Wyh;
dLpds = dsm;

% Within time step
dh = dLdh;
ds = dLpds;
do = dh.*tanhs';

dtanhs = dh.*o';
ds = ds + dtanhs.*(1-tanhs.^2)';

dg = ds.*ii';
di = ds.*g';
dsm = ds.*f';
df = ds.*sm';

dglin = dg.*(1-g.^2)';
dilin = di.*(ii.*(1-ii))';
dflin = df.*(f.*(1-f))';
dolin = do.*(o.*(1-o))';
dgifo_lin = [dglin,dilin,dflin,dolin];
dW_gifo_x = dW_gifo_x + dgifo_lin'*x(:,t:t+batchSize-1)';
dW_gifo_h = dW_gifo_h + dgifo_lin'*hm';
db_gifo = db_gifo + sum(dgifo_lin,1)';
dhm = dgifo_lin * W_gifo_h;

