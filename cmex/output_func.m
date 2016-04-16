toc(timeCount);
% Compute loss for training set and validation set
loss_training = loss_thread(:,:,1:end-1);
loss_validation = loss_thread(:,:,end);
% Collapse w.r.t periods
loss_training = sum(loss_training,1);
loss_validation = sum(loss_validation,1);
% Collapse across batches
loss_training = mean(loss_training(:));
loss_validation = mean(loss_validation(:));

% Update learning Rate if no improvement
if loss_training > last_loss_training
    learningRate = learningRate * learningRateDecay;
end
last_loss_training = loss_training;

fprintf('Saving at %d iters, with %d samples processed\n',saveCount,currentBatch*batchSize);
save(['weights_' num2str(saveCount) '_' num2str(currentBatch*batchSize) '_' num2str(loss_training) '_' num2str(loss_validation) '.mat'], 'weights');
timeCount = tic;