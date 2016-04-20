function Asimov_gen
weights = load('weights_10_320000_171.532_363.7563.mat');
weights = v2struct(weights);
numWordsPredicted = 2e3;

text = fileread('shaofubaijie.txt');
[text_dic,code_text,text_code] = unique(text);

% Initial words
% words = fileread('start_words.txt');
words = text(1000:1100);
[~,words_code] = ismember(words,text_dic);
words_code_linear = zeros(max(text_code),length(words_code));
words_code_linear(1:max(words_code),:) = dummyvar(words_code)';

batchSize = 64;
periods = 100;
hDims = [256 256];
nLayer = 2;

xDim = size(words_code_linear,1);
yDim = size(words_code_linear,1);

NumThreads = 4;
NSlots = getenv('NSLOTS');
if (strcmp(NSlots, '') == 0)
    NumThreads = str2double(NSlots);
    fprintf('NumThreads: %d\n',NumThreads);
end

dropoutRate = 0.5;
typename = 'single';
nnetName = ['lstmNet_' typename];

params = v2struct(xDim,yDim,batchSize,periods,nLayer,hDims,NumThreads,dropoutRate,typename);


%% Predict
addpath('../cmex');
clear(nnetName);
tic;
for j=1:numWordsPredicted
    xData = words_code_linear(:,end-periods+1:end);
    yhat = lstm_predict(xData,[],nnetName,params,weights);
    [~,iymax] = max(yhat(:,1,end));
    predict = zeros(size(words_code_linear,1),1);
    predict(iymax) = 1;
    words_code_linear = [words_code_linear predict];
%     words_code_linear = [words_code_linear yhat(:,end)];
    words_code = words_code_linear'*[1:size(words_code_linear,1)]';
    words = text_dic(words_code);
%     words = [words text_dic(iymax)];
end
toc;
fileId = fopen('generated_txt.txt','w');
fprintf(fileId,'%s',words);
fclose(fileId);
end