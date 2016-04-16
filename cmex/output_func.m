toc(timeCount);
fprintf('Saving at %d iters, with %d samples processed\n',saveCount,currentBatch*batchSize);
save(['weights_' num2str(saveCount) '_' num2str(currentBatch*batchSize)],'weights');
timeCount = tic;