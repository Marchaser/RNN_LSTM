toc(timeCount);
fprintf('Saving at %d iters, with %d samples processed\n',saveCount,batchStart-1);
save(['weights_' num2str(saveCount) '_' num2str(batchStart-1)],'weights');
timeCount = tic;