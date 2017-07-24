close all; clear all; 
% clc;
addpath('./utils/');
addpath MDH\;
addpath ..\code;

%db_name = 'cifar_10_gist';
%db_name = 'AR_Maxminnorm_caffe';
%db_name = 'FLICKR_25k_caffe_MaxminNorm';
%db_name = 'gist_512d_Caltech-256';
% db_name = 'cifar_10_norm'
% db_name = 'cifar_10_norm_emdedded'
 db_name = 'cifar_10_norm_embedded_zeroshot'
% db_name = 'cifar_10_norm_zeroshot'

options.choice = 'evaluation';

loopnbits = [32];
runtimes = 1;    % modify it more times such as 8 to make the rusult more precise

choose_times = 1;    % k is the times of run times to show for evaluation
options.zeroshot=0;

% load dataset
if strcmp(db_name, 'AR_Maxminnorm_caffe')
    load AR_Maxminnorm_caffe.mat;
    
elseif strcmp(db_name, 'cifar_10_gist')
    load cifar_10_gist.mat;
    
elseif strcmp(db_name, 'FLICKR_25k_caffe_MaxminNorm')
    load FLICKR_25k_caffe_MaxminNorm.mat;
    
elseif strcmp(db_name, 'cnn_1024d_Caltech-256')
    load cnn_1024d_Caltech-256.mat;
    
elseif strcmp(db_name, 'cifar_10_norm')
    load cifar_10_norm;
    
elseif strcmp(db_name, 'cifar_10_norm_emdedded')
    load cifar_10_norm_emdedded;
    
elseif strcmp(db_name, 'cifar_10_norm_embedded_zeroshot')
    load cifar_10_norm_embedded_zeroshot;
    options.zeroshot = 1
elseif strcmp(db_name, 'cifar_10_norm_zeroshot')
    load cifar_10_norm_zeroshot;
    options.zeroshot = 1
end

hashmethods = {'MDH'};    % CBE training process is very slow
nhmethods = length(hashmethods);

result.mAP = zeros(nhmethods,length(loopnbits));
result.Precision = zeros(nhmethods,length(loopnbits));
result.Recall = zeros(nhmethods,length(loopnbits));
result.F1 = zeros(nhmethods,length(loopnbits));


profile on

options.Ntrain = min(10000,size(traindata,1));
options.n_anchors = 1000;
options.nu = 1e-5;
options.epsilon = options.nu*1e3; % overfitting
options.lambda = 1e-2;
options.delta = 1e-2; % not sensitive????
options.k = 5;
options.tol = 1e-3;

%code length
options.l = 64;

% options.rotate = 0 is SDH, maxItr should be set to 3.
% options.rotate = 1 && orth = 1, R'*R=I, maxItr should be set to 10.
options.maxItr = 10;
options.orth = 1;
options.rotate = 0;

rand('seed',1);
ix = randsample(size(traindata,1), options.Ntrain);
X = traindata(ix,:);
if exist('traingnd','var')
    label = double(traingnd(ix,:));
elseif exist('Y_train','var')
    label = double(traingnd(ix,:));
end
anchor = X(randsample(options.Ntrain, options.n_anchors),:);
sigma = mean(mean(sqdist(X,anchor)))/2;
PhiX = exp(-sqdist(X,anchor)/(2 *sigma));
PhiX = [PhiX, ones(options.Ntrain,1)];
Phi_testdata = exp(-sqdist(testdata,anchor)/(2  * sigma));
Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
Phi_traindata = exp(-sqdist(traindata,anchor)/(2  *sigma));
Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];

[~, P,~,R] = mdh(PhiX',label',options);

if options.zeroshot
    FX = [Phi_traindata;Phi_testdata(1:5000,:)]*P;
    tFX = Phi_testdata(5001:end,:)*P;
else
    FX = Phi_traindata*P;
    tFX = Phi_testdata*P;
end
B = FX > 0;
tB = tFX > 0;

[result.mAP, result.Precision, result.Recall, result.F1] = evaluation(B,tB,cateTrainTest);
result
profile report