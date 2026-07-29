function R_test = Enhanced_cov_test(X, K, tau, Wh)
% Compute enhanced covariace matrices of test set
%
% Inputs :
% X         : EEG signals of test set
% K         : Order of FIR filter
% tau       : Time delay parameter
% Wh        : Whitening matrix for enhancing covariance matrices

% Outputs   :
% R_test    : Enhanced covariace matrices of test set
% ************************************************************************
[C, T, M] = size(X);
KC = K*C; % [KC, KC]: dimension of augmented covariance matrix
Cov = cell(1, M);
Sig_Cov = zeros(KC, KC);
for m = 1:M
    X_m = X(:,:,m);
    X_m_hat = [];
    % Generate augumented EEG data
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
    % Compute covariance and trace normalization
    Cov{1,m} = X_m_hat*X_m_hat';
    Cov{1,m} = Cov{1,m}/trace(Cov{1,m});
    Sig_Cov = Sig_Cov + Cov{1,m};
end

% Whitenning, logarithm transform, and vectorization
Cov_whiten = zeros(M, KC, KC);
for m = 1:M
    temp_cov = Wh^(-1/2)*Cov{1,m}*Wh^(-1/2);
    Cov_whiten(m,:,:) = (temp_cov + temp_cov')/2;
    R_m =logm(squeeze(Cov_whiten(m,:,:))); % logarithm transform
    R_m = R_m(:); % column-wise vectorization
    R_test(m,:) = R_m';
end
end