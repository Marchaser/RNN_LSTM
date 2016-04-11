% Backward propagation for one time step
% Back through time
dyhat = dfuncLoss(y,yhat);
dWyh = dWyh + dyhat(:)*h(:)';
dby = dby + dyhat(:);
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

dhm = 0;

dolin = do.*(o.*(1-o))';
dhm = dhm + dolin*Woh;
dWox = dWox + dolin(:)*x(:)';
dWoh = dWoh + dolin(:)*hm(:)';
dbo = dbo + dolin(:);

dflin = df.*(f.*(1-f))';
dhm = dhm + dflin*Wfh;
dWfx = dWfx + dflin(:)*x(:)';
dWfh = dWfh + dflin(:)*hm(:)';
dbf = dbf + dflin(:);

dilin = di.*(ii.*(1-ii))';
dhm = dhm + dilin*Wih;
dWix = dWix + dilin(:)*x(:)';
dWih = dWih + dilin(:)*hm(:)';
dbi = dbi + dilin(:);

dglin = dg.*(1-g.^2)';
dhm = dhm + dglin*Wgh;
dWgx = dWgx + dglin(:)*x(:)';
dWgh = dWgh + dglin(:)*hm(:)';
dbg = dbg + dglin(:);

