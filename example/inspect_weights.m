%
clear;

weights1 = load('weights_5_1000_64000_104.5839_396.5334.mat');
weights1 = weights1.weights;
tic;
sum(abs(weights1))
toc;
% figure;
% hist(weights);

weights2 = load('weights_10_2000_128000_74.4662_468.6375.mat');
weights2 = weights2.weights;
tic;
sum(abs(weights2))
toc;
% figure;
% hist(weights);