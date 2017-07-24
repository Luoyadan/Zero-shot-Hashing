function [ap] = cat_apcal(cateTrainTest, IX)
% ap=apcal(score,label)
% average precision (AP) calculation 

[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
for i = 1 : numtest
    y = IX(:,i);
    x=0;%number of retrieved samples
    p=0;%number of sum precision
    new_label=cateTrainTest(:,i);

    
    num_return_NN = min(numtrain,10000); % only compute MAP on returned top 1000 neighbours.
    for j=1:num_return_NN
        if new_label(y(j))==1
            x=x+1;
            p=p+x/j;
        end
    end  
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end
    
    
end

ap = mean(apall);
