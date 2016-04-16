function Asimov_gen
weights = load('weights_2000_128000_323.4194_320.7245.mat');
weights = v2struct(weights);

text = fileread('Foundation - Isaac Asimov.txt');
[text_dic,code_text,text_code] = unique(text);

% Initial words
% words = fileread('start_words.txt');
words = text(1000:1100);
[~,words_code] = ismember(words,text_dic);
words_code_linear = zeros(max(text_code),length(words_code));
words_code_linear(1:max(words_code),:) = dummyvar(words_code)';

batchSize = 100;
periods = 100;
hDims = [256 256];
nLayer = 2;

xDim = size(words_code_linear,1);
yDim = size(words_code_linear,1);
NumThreads = 4;

params = v2struct(xDim,yDim,batchSize,periods,nLayer,hDims,NumThreads);

%% Predict
addpath('../cmex');
numWordsPredicted = 5e2;
clear lstmNet;
for j=1:numWordsPredicted
    xData = words_code_linear(:,end-periods+1:end);
    yhat = lstm_predict(xData,'lstmNet',params,weights);
    [~,iymax] = max(yhat(:,1,end));
    predict = zeros(size(words_code_linear,1),1);
    predict(iymax) = 1;
    words_code_linear = [words_code_linear predict];
%     words_code_linear = [words_code_linear yhat(:,end)];
    words_code = words_code_linear'*[1:size(words_code_linear,1)]';
    words = text_dic(words_code);
%     words = [words text_dic(iymax)];
end
fileId = fopen('generated_txt.txt','w');
fprintf(fileId,'%s',words);
end