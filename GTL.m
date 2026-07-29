function [max_acc,Cls,Obj,U,Vs,Vt] = GTL(Xs,Xt,Ys,Yt,options)

% Mingsheng Long et al. Transfer Learning with Graph Co-Regularization. TKDE 2012.

if nargin < 5
    error('No algorithm parameters provided!');
end
if ~isfield(options,'p')
    options.p = 10;
end
if ~isfield(options,'lambda')
    options.lambda = 0.1;
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'sigma')
    options.sigma = 10.0;
end
if ~isfield(options,'iters')
    options.iters = 200;
end
if ~isfield(options,'data')
    options.data = 'default';
end
p = options.p;
lambda = options.lambda;
gamma = options.gamma;
sigma = options.sigma;
iters = options.iters;
data = options.data;

fprintf('GTL: data=%s  p=%d  lambda=%f  gamma=%f  sigma=%f\n',data,p,lambda,gamma,sigma);

%% Set predefined variables (Yt only for test)
X = [Xs,Xt];
Y = [Ys;Yt];
m = size(X,1);
c = length(unique(Y));
ns = size(Xs,2);
nt = size(Xt,2);
YY = [];
for i = reshape(unique(Y),1,length(unique(Y)))
    YY = [YY,Y==i];
end
[~,Y] = max(YY,[],2);
Ys = YY(1:ns,:);
Yt = YY(ns+1:end,:);

%% Data normalization (for classification)
Xs = Xs*diag(sparse(1./sqrt(sum(Xs.^2))));
Xt = Xt*diag(sparse(1./sqrt(sum(Xt.^2))));

%% Construct graph Laplacian
manifold.k = p;
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
manifold.bNormalizeGraph = 0;
 [Wus,Dus] = laplacian(Xs,manifold);
[Wut,Dut] = laplacian(Xt,manifold);
[Wvt,Dvt] = laplacian(Xt',manifold);
% manifold.NeighborMode = 'Supervised';
% [~,manifold.gnd] = max(Ys,[],2);
% [Wvs,Dvs] = laplacian(Xs',manifold);

%% Initialization
U = rand(m,c);
Vs = 0.1 + 0.8*Ys;
if isfield(options,'Yt0') && size(options.Yt0,1)==nt
    Vt = [];
    for i = reshape(unique(options.Yt0),1,length(unique(options.Yt0)))
        Vt = [Vt,options.Yt0==i];
    end
    options.Yt0 = [];
    Vt = 0.1 + 0.8*Vt;
else
    Vt = rand(nt,c);
end

%% Graph Co-Regularized Transfer Learning (GTL)
Acc = [];
Obj = [];
for it = 0:iters
    
    %% Alternating Optimization
    if it>0
        U = U.*sqrt((Xs*Vs+Xt*Vt+lambda*Wus*U+lambda*Wut*U)./(U*(Vs'*Vs)+U*(Vt'*Vt)+lambda*Dus*U+lambda*Dut*U+eps));
        
        Vs = Vs.*sqrt(Vs./(Vs*(Vs'*Vs)+eps));

        Vt = Vt.*sqrt((Xt'*U+gamma*Wvt*Vt+sigma*Vt)./(Vt*(U'*U)+gamma*Dvt*Vt+sigma*Vt*(Vt'*Vt)+eps));
    end
    
    %% Compute accuracy
    [~,Cls] = max(Vt,[],2);
    [~,Lbl] = max(Yt,[],2);
    acc = numel(find(Cls == Lbl))/nt;
    if it>1 && acc<max_acc
        break;
    else
        max_acc = acc;
    end
    
    %% Compute objective
    O = 0;
%     % Comment for fast evaluations
%     O = norm(Xs-U*Vs','fro')^2 + norm(Xt-U*Vt','fro')^2 ...
%         + sigma*norm(Vs'*Vs-eye(c,c),'fro')^2 + sigma*norm(Vt'*Vt-eye(c,c),'fro')^2 ...
%         + lambda*trace(U'*(Dus-Wus)*U) + lambda*trace(U'*(Dut-Wut)*U) ...
%         + gamma*trace(Vs'*(Dvs-Wvs)*Vs) + gamma*trace(Vt'*(Dvt-Wvt)*Vt);
    Obj = [Obj;O];
    
    if mod(it,10)==0
        fprintf('[%d]  objective=%0.10f  accuracy=%0.4f\n',it,O,acc);
    end
end

fprintf('Algorithm GTL terminated!!!\n\n\n');

end