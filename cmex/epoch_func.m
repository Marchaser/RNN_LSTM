fprintf('------------------------------------------\n');
epochCount = epochCount + 1;
fprintf('Current epoch: %d\n',epochCount);
fprintf('Time used in training: %g\n',toc(timeTic));
% Store loss in training
lossTraining = loss_thread;
lossTraining = sum(lossTraining,1);
lossTraining = mean(lossTraining(:));

timeTic = tic;
MEX_TASK = MEX_INIT_MEMORY; % Clear memory
eval(netMexName);
MEX_TASK = MEX_PREDICT_BATCH;
lossValidation = zeros(size(loss_thread));
batchValidationStride = batchValidationStride0;
% batchValidationStride = batchTrainingStride;
for validationStep=1:lengthValidationBatch
    batchDataStride = batchValidationStride;
    eval(netMexName);
    % Accumulate validation error
    lossValidation = lossValidation + loss_thread;
    batchValidationStride = batchValidationStride + 1;
end
% Collapse w.r.t periods
lossValidation = sum(lossValidation,1);
% Collapse across batches
lossValidation = sum(lossValidation(:)) / (lengthValidationBatch*batchSize);
fprintf('Time used in validation: %g\n',toc(timeTic));
fprintf('Total time used: %g\n',toc(totalTimeTic));

fprintf('lossTraining: %g\nlossValidation: %g\n',lossTraining,lossValidation);
fprintf('Max abs(weights): %g\n',max(abs(weights)));

% Update learning Rate if no improvement
%{
if loss_training > last_loss_training
    learningRate = learningRate * learningRateDecay;
end
last_loss_training = loss_training;
%}
trainInfo.lossTraining = [trainInfo.lossTraining lossTraining];
trainInfo.lossValidation = [trainInfo.lossValidation lossValidation];
fprintf('Saving at %d epoch, with %d samples trained\n',epochCount,currentTotalSamples);
fprintf('------------------------------------------\n');
save(['weights_' num2str(epochCount) '_' num2str(step) '_' num2str(currentTotalSamples) '_' num2str(lossTraining) '_' num2str(lossValidation) '.mat'], 'weights');
save(['trainInfo'], 'trainInfo');
MEX_TASK = MEX_INIT_MEMORY; % Clear memory
eval(netMexName);
% Count time forward
timeTic = tic;