function Asimov_gen
weights = load('weights_2700_172800');
weights = v2struct(weights);

text = fileread('t8.shakespeare.txt');
text = text(1:1e6);
[text_dic,code_text,text_code] = unique(text);

% Initial words
words = text(1000:1100);
[~,words_code] = ismember(words,text_dic);
words_code_linear = zeros(max(text_code),length(words_code));
words_code_linear(1:max(words_code),:) = dummyvar(words_code)';

temperature = 1;
batchSize = 100;
learningRate = 0.01;
periods = 100;
hDim = 256;
hDim1 = 256;
hDim2 = 256;

yDim = size(words_code_linear,1);
NumThreads = 4;

params = v2struct(temperature,batchSize,learningRate,periods,hDim,hDim1,hDim2,yDim,NumThreads);

netName = 'twoLayerNet';

%% Predict
addpath('../cmex');
numWordsPredicted = 2e2;
clear(netName);
for j=1:numWordsPredicted
    xData = words_code_linear(:,end-periods+1:end);
    yhat = lstm_predict(xData,netName,params,weights);
    [~,iymax] = max(yhat(:,1,end));
    predict = zeros(size(words_code_linear,1),1);
    predict(iymax) = 1;
    words_code_linear = [words_code_linear predict];
%     words_code_linear = [words_code_linear yhat(:,end)];
    words_code = words_code_linear'*[1:size(words_code_linear,1)]';
    words = text_dic(words_code);
%     words = [words text_dic(iymax)];
end
end