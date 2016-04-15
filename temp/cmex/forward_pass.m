% Compute forwardpass one time
gifo_lin = gifo_x_t(:,t:t+batchSize-1) + W_gifo_h*hm + repmat(b_gifo,1,batchSize);
glin = gifo_lin(0*hDim+1:1*hDim,:);
g = tanh(glin);
ilin = gifo_lin(1*hDim+1:2*hDim,:);
ii = sigmoid(ilin);
flin = gifo_lin(2*hDim+1:3*hDim,:);
f = sigmoid(flin);
olin = gifo_lin(3*hDim+1:4*hDim,:);
o = sigmoid(olin);

s = g.*ii + sm.*f;
tanhs = tanh(s);
h = tanhs.*o;

% Layer output to final output
ylin = Wyh*h + repmat(by,1,batchSize);
ylin_max = max(ylin,[],1);
exp_ylin_minus_max = exp(ylin - repmat(ylin_max,yDim,1));
sum_exp_ylin_minus_max = sum(exp_ylin_minus_max,1);
yhat = exp_ylin_minus_max ./ repmat(sum_exp_ylin_minus_max,yDim,1);