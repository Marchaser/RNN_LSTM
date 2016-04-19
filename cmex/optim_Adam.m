% Adam, input(weights,dweights) output(dweights)
Adam_m = Adam_beta1*Adam_m + (1-Adam_beta1)*dweights;
Adam_v = Adam_beta2*Adam_v + (1-Adam_beta2)*dweights.*dweights;
Adam_mhat = Adam_m/(1-Adam_beta1^t);
Adam_vhat = Adam_v/(1-Adam_beta2^t);
weights = weights - learningRate*Adam_mhat./(Adam_vhat.^0.5+Adam_epsilon);