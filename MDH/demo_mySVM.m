function [B,tB] = demo_mySVM(traindata,testdata,traingnd,testgnd,param)

train=param.Ntrain;
codeLen=param.nbits;
for Ntrain = train,
    rand('seed',1);
    ix = randsample(size(traindata,1), Ntrain);
    X = traindata(ix,:);
    if exist('traingnd','var')
        label = double(traingnd(ix,:));
    elseif exist('Y_train','var')
        label = double(traingnd(ix,:));
    end
    
    % get anchors
    n_anchors = 1000;
    % rand('seed',1);
    anchor = X(randsample(Ntrain, n_anchors),:);
    sigma =mean(mean(sqdist(X,anchor)))/2;
    PhiX = exp(-sqdist(X,anchor)/(2 *sigma));
    PhiX = [PhiX, ones(Ntrain,1)];
    % sigma =mean(mean(sqdist(testdata,anchor)))/2;
    Phi_testdata = exp(-sqdist(testdata,anchor)/(2  * sigma));
    Phi_testdata = [Phi_testdata, ones(size(Phi_testdata,1),1)];
    % sigma =mean(mean(sqdist(traindata,anchor)))/2;
    Phi_traindata = exp(-sqdist(traindata,anchor)/(2  *sigma));
    Phi_traindata = [Phi_traindata, ones(size(Phi_traindata,1),1)];
    
    % learn G and F
    maxItr = 10;
    gmap.lambda = 1; gmap.loss = 'L2';
    Fmap.type = 'RBF'; Fmap.lambda = 1e-2; Fmap.sigmoid = false;
    Fmap.nu = 1e-5; %  penalty parm for F, 1e-5 for mySVM_binary, 1e-1 for mySVM;
    Rmap.delta = 1e-2;
    
    nbits = codeLen; % low dimension L in the method
    
    % Init Z
    randn('seed',3);
    Zinit=sign(randn(Ntrain,nbits));
    debug = 1;
    %                             addpath code\;
    %                             S = constructW(PhiX);
    %                             L = diag(sum(S))-S;
    %                             L1 = ones(size(L));
    %                             [R,~,D]=svd(label'*L1*label);
    %                             label1 = label*R;
    
    [~, F, ~] = mySVM_binary(PhiX,label,Zinit,gmap,Fmap,Rmap,[],maxItr,debug);
    
    
    if param.zeroshot
        FX = [Phi_traindata ;Phi_testdata(1:5000,:)]*F.W;
        tFX = Phi_testdata(5001:end,:)*F.W;
    else
        FX = Phi_traindata*F.W;
        tFX = Phi_testdata*F.W;
    end
    
    B = FX > 0;
    tB = tFX > 0;
    
end
end




