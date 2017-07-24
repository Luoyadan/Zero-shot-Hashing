function [mAP, Precision, Recall, F1] = evaluation(H,tH,cateTrainTest)
%EVALUATION 此处显示有关此函数的摘要
%   此处显示详细说明
    %display('...Evaluation...');
    hammRadius = 2;
    addpath('./eval');
    

    B = compactbit(H);
    tB = compactbit(tH);
%    num_test = size(tB,1);
%____________Evaluation____________
    hammTrainTest = hammingDist(tB, B)';
     % hash lookup
    Ret = (hammTrainTest <= hammRadius+0.00001);
    [Precision, Recall] = evaluate_macro(cateTrainTest, Ret);
    % [cateP(ix_len), cateR(ix_len)] = evaluate_macro(trueTrainTest, Ret)
    % get hamming ranking: MAP, precision and reall
    [~, HammingRank]=sort(hammTrainTest,1);
    mAP = cat_apcal(cateTrainTest,HammingRank);%compute retrival mAP on all training data.
    F1 = F1_measure(Precision, Recall);

end

