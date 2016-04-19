% RmsProp, input(weights, dweights), output(weights)
RmsProp_r = (1-RmsPropDecay)*dweights.^2 + RmsPropDecay*RmsProp_r;
RmsProp_v = learningRate ./ (RmsProp_r.^0.5);
RmsProp_v = min(RmsProp_v,learningRate*10);
RmsProp_v = max(RmsProp_v,learningRate/10);
RmsProp_v = RmsProp_v.* dweights;

weights = weights - RmsProp_v;