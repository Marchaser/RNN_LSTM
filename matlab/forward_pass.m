% Compute forwardpass one time
glin = Wgx*x + Wgh*hm + bg;
g = tanh(glin);
ilin = Wix*x + Wih*hm + bi;
ii = sigmoid(ilin);
flin = Wfx*x + Wfh*hm + bf;
f = sigmoid(flin);
olin = Wox*x + Woh*hm + bo;
o = sigmoid(olin);

s = g.*ii + sm.*f;
tanhs = tanh(s);
h = tanhs.*o;

% Layer output to final output
ylin = Wyh*h + by;
yhat = exp(ylin-max(ylin)) / sum(exp(ylin-max(ylin)));