function [x_hat] = temporalReg(x,K,tau)
%TEMPORALREG 此处显示有关此函数的摘要
%   此处显示详细说明
    T = size(x,2);
    for tr = 1:size(x,3)
        X_m = squeeze(x(:,:,tr));       
        X_m_hat = [];
        for k = 1:K
            n_delay = (k-1)*tau;
            if n_delay ==0
                X_order_k = X_m;
            else
                X_order_k(:,1:n_delay) = 0;
                X_order_k(:,n_delay+1:T) = X_m(:,1:T-n_delay);
            end
            X_m_hat = cat(1,X_m_hat,X_order_k);
        end
        x_hat(:,:,tr) = X_m_hat;
    end
end

