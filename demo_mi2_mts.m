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
% 9 subjects, each 22*750*144 (channels*points*trails)
root='.\MI2-1\';
listing=dir([root '*.mat']);
addpath('lib');

% Load data and perform congruent transform
fnum=length(listing);
Ca=nan(22,22,144*fnum);
Xr=nan(22,750,144*9);
Xa=nan(22,750,144*9);
La=nan(22,750,144*9);
Y=nan(144*fnum,1);
ref={'riemann','logeuclid','euclid'};
for f=1:fnum
    load([root listing(f).name])
    idf=(f-1)*144+1:f*144;
    Y(idf) = y; Xr(:,:,idf) = x;
    Ca(:,:,idf) = centroid_align(x,ref{2});
%     [~,La(:,:,idf)] = centroid_align(x,ref{2});
    [~,Xa(:,:,idf)] = centroid_align(x,ref{2});
end

% La = covariances(La);
% Xa = covariances(Xa);
    
N=1; bca_dte=[];
for t=1:N

    BCA=zeros(fnum,1);
    for n=1:fnum
        disp(n)
        % Single target data & multi source data
        idt=(n-1)*144+1:n*144;
        ids=1:144*fnum; ids(idt)=[];             
        Yt=Y(idt); Ys=Y(ids);
        idsP=Yt==1; idsN=Yt==0;
        
       %% Logarithmic mapping on aligned covariance matrices
        % Ct=Ca(:,:,idt);  Cs=Ca(:,:,ids);
        % Xs=logmap(Cs,'MI'); % dimension: 253*1152 (features*samples)
        % Xt=logmap(Ct,'MI');

        %%Temporally regularized%%
        Ft = Xa(:,:,idt);  Fs = Xa(:,:,ids);
        K = 2; tau = 5; 
        train_x_hat= temporalReg(Fs, K, tau);
        test_x_hat = temporalReg(Ft, K, tau);
        [Xs,Xt] = TSfeature_Cov(train_x_hat, test_x_hat);
        

        % [Xs2, Xt2] = CSPfeature(Fs, Ys, Ft, 10);
        % Xs = [Xs1;Xs2'];
        % Xt = [Xt1;Xt2'];
        % K = 2;
        % tau = 5;
        % [R_train, Wh] = Enhanced_cov_train(Fs, K, tau);
        % R_test = Enhanced_cov_test(Ft, K, tau, Wh);
        % Xs = R_train';
        % Xt = R_test';
        
        
        %% MEKT
%         options.d = 10;             % subspace bases 
%         options.T = 5;              % iterations, default=5
%         options.alpha= 0.01;        % the parameter for source discriminability
%         options.beta = 0.1;         % the parameter for target locality, default=0.1
%         options.rho = 20;           % the parameter for subspace discrepancy
%         options.clf = 'slda';        % the string for base classifier, 'slda' or 'svm'
%         Cls = [];
%         [Zs, Zt] = MEKT(Xs, Xt, Ys, Cls, options);
%         Ypre = slda(Zt,Zs,Ys);
%         BCA(n)=.5*(mean(Ypre(idsP)==1)+mean(Ypre(idsN)==0));
%         clear options
 
       %% Co-Graph
        algorithm = 'GCMF';             % 'GTL' | 'GCMF' | 'GTL3'

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
            options.p = 10;             % insensitive, keep default
            options.lambda = 0.1;       % insensitive, keep default
            options.gamma = 4.0;       % 1<=gamma<=100
            options.sigma = 10.0;      % gamma<=sigma<=10*gamma
        else
            error('Unsupported algorithm!\n');
        end
        options.iters = 30;
        w=ones(size(Ys)); w(Ys==1)=sum(Ys==0)/sum(Ys==1);
        Yt0 = slda(Xt,Xs,Ys);
%         LDA = fitcdiscr(Xs',Ys);
%         Yt0=predict(LDA,Xt');
        options.Yt0 = Yt0;
        acc = feval(algorithm,Xs,Xt,Ys,Yt,options);
        BCA(n) = acc;
        clear options

 
    end
    disp(mean(BCA)*100)
    bca_dte=[bca_dte,mean(BCA)*100];
end

rmpath('lib');
