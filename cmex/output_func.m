toc(timeCount);
fprintf('Saving at %d iters, with %d samples processed\n',saveCount,batchEnd);
save(['weights_' num2str(saveCount) '_' num2str(batchEnd)],'W_gifo_x','W_gifo_h','b_gifo','Wyh','by');
timeCount = tic;