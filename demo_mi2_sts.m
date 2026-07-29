% =====================
% Manifold Embedded Knowledge Transfer for Brain-Computer Interfaces (MEKT)
% =====================
% Author: Wen Zhang and Dongrui Wu
% Date: Oct. 9, 2019
% E-mail: wenz@hust.edu.cn

clc;
clear all;
close all;
warning off;

% Load datasets:
% 7 subjects, each 59*300*200 (channels*points*trails)
root='MI2-1\';
listing=dir([root '*.mat']);
addpath('lib');

fnum=length(listing);
BCA=zeros(fnum,fnum-1);
ref={'riemann','logeuclid','euclid'};
tic
for tr=1:fnum
    disp(tr)
    % Single target data
    load([root listing(tr).name])
    Xtr=x; Yt=y;
    tes=1:fnum; tes(tr)=[];
    
    for te=1:fnum-1
        % Single source data
        load([root listing(tes(te)).name])
        Xsr=x; Ys=y;
        % idsP=Yt==1; idsN=Yt==-1;
        
        % Centroid Alignment
%         Cs=centroid_align(Xsr,ref{3});
%         Ct=centroid_align(Xtr,ref{3});
% 
%         
%         % Logarithmic mapping on aligned covariance matrices

        [Cs,Fs] = centroid_align(Xsr,ref{2});
        [Ct,Ft] = centroid_align(Xtr,ref{2});
        % Logarithmic mapping on aligned covariance matrices
        % Xs=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
        % Xt=logmap(Ct,'MI');


        %%Temporally regularized%%
        K = 2; tau = 4; 
        % train_x_hat = temporalAug(Fs, K, tau);
        % test_x_hat = temporalAug(Ft, K, tau);
        train_x_hat = temporalReg(Fs, K, tau);
        test_x_hat = temporalReg(Ft, K, tau);
        % [R_train, Wh] = Enhanced_cov_train(Fs, K, tau);
        % R_test = Enhanced_cov_test(Ft, K, tau, Wh);
        % Xs = R_train';  Xt = R_test'; 
        [Xs,Xt] = TSfeature_Cov(train_x_hat, test_x_hat);

        %% Co-Graph
        algorithm = 'GTL';             % 'GTL' | 'GCMF' | 'GTL3'

        if strcmp(algorithm,'GTL')
            options.p = 10;             % insensitive, keep default
            options.lambda = 0.1;       % insensitive, keep default
            options.gamma = 2.0;       % 1<=gamma<=100
            options.sigma = 10.0;      % gamma<=sigma<=10*gamma
        elseif strcmp(algorithm,'GCMF')
            options.k = 10;             % insensitive, keep default
            options.p = 10;             % insensitive, keep default
            options.lambda = 0.1;       % 0<=lambda<=1.0
            options.gamma = 100.0;      % 10<=gamma<=1000
        elseif strcmp(algorithm,'GTL3')
            options.p = 5;             % insensitive, keep default
            options.lambda = 0.1;       % insensitive, keep default
            options.gamma = 100.0;       % 1<=gamma<=100
            options.sigma = 1000.0;      % gamma<=sigma<=10*gamma
        else
            error('Unsupported algorithm!\n');
        end
        options.iters = 30;

        Yt0 = slda(Xt,Xs,Ys);
        options.Yt0 = Yt0;
        Acc = feval(algorithm,Xs,Xt,Ys,Yt,options);
        BCA(tr, te) = max(Acc);
        clear options

        % K = 2;
        % tau = 1;
        % [R_train, Wh] = Enhanced_cov_train(Fs, K, tau);
        % R_test = Enhanced_cov_test(Ft, K, tau, Wh);
        % Xs = R_train';
        % Xt = R_test';
        %% DLAD
        % options.r = 10;
        % options.eta = 0.1;
        % options.lambda = 50;
        % options.T = 5;
        % [Acc,acc_iter,Yt_pred] = DLAD(Xs,Ys,Xt,Yt,options);
        % BCA(tr,te)=Acc;
        
%         Xs2 = logmap(Cs,'MI'); % dimension: 1770*200 (features*samples)
%         Xt2 = logmap(Ct,'MI');
%         
%         Xs = [Xs1; Xs2];
%         Xt = [Xt1; Xt2];
        % options= defaultOptions(struct(),...
        % 'T',5,...              % The iteration times
        % 'dim',10,...            % The dimension of the projection subspace
        % 'alpha',0.1,...         % The weight of manifold regularization
        % 'beta',5,...            % The weight of discrimination
        % 'sC',2,...             % The fuzzy number
        % 'kernel_type',1,...     % Kernel
        % 'gamma',1,...           % The hyper-parameter of Kernel
        % 'lambda',1);            % The regularization term
        % 
        % [acc,acc_ite,max_acc]=SCSL(Xs1,Ys,Xt1,Yt,options);
        % BCA(tr,te) = max_acc;
        
        %% MEKT
%         options.d = 100;             % subspace bases 
%         options.T = 5;              % iterations, default=5
%         options.alpha= 0.01;        % the parameter for source discriminability
%         options.beta = 0.1;         % the parameter for target locality, default=0.1
%         options.rho = 20;           % the parameter for subspace discrepancy
%         options.clf = 'slda';        % the string for base classifier, 'slda' or 'svm'
%         Cls = [];
%         [Zs, Zt] = MEKT(Xs1, Xt1, Ys, Cls, options);
%         Ypre = slda(Zt,Zs,Ys);
%         BCA(tr,te)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==-1));
       
    end
end
toc
disp(mean(mean(BCA,1),2)*100')

rmpath('lib');
