% Adam, input(weights,dweights) output(dweights)
Adam_m = Adam_beta1*Adam_m + (1-Adam_beta1)*dweights;
Adam_v = Adam_beta2*Adam_v + (1-Adam_beta2)*dweights.*dweights;
factor1 = 1/(1-Adam_beta1^step);
factor2 = 1/(1-Adam_beta2^step);
Adam_mhat = Adam_m*factor1;
Adam_vhat = Adam_v*factor2;
weights = weights - learningRate*Adam_mhat./(Adam_vhat.^0.5+Adam_epsilon);