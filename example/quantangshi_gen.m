function quantangshi_gen
weights = load('weights_78_15600_998400_277.3535_306.0343.mat');
weights = v2struct(weights);
numWordsPredicted = 2e3;

max(weights)
min(weights)

text = fileread('quantangshi.txt');
text = text(1:1e6);
[text_dic,code_text,text_code] = unique(text);

% Initial words
% words = fileread('start_words.txt');
words = text(1001:1101);
[~,words_code] = ismember(words,text_dic);
words_code_linear = zeros(max(text_code),length(words_code));
words_code_linear(1:max(words_code),:) = dummyvar(words_code)';

batchSize = 64;
periods = 100;
hDims = [512 512];
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