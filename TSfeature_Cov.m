function [Xs,Xt] = TSfeature_Cov(xTrain,xTest)
    for i=1:size(xTrain,3)
        Cs(:,:,i)=cov(xTrain(:,:,i)');
    end
    
    for i=1:size(xTest,3)
        Ct(:,:,i)=cov(xTest(:,:,i)');
    end
    
    Xs=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
    Xt=logmap(Ct,'MI');
    
end

