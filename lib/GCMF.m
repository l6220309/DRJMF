function [max_acc,Cls,Obj,Us,Ut,H,Vs,Vt] = GCMF(Xs,Xt,Ys,Yt,options)

% Mingsheng Long et al. Transfer Learning with Graph Co-Regularization. AAAI 2012.

if nargin < 5
    error('No algorithm parameters provided!');
end
if ~isfield(options,'k')
    options.k = 10;
end
if ~isfield(options,'p')
    options.p = 10;
end
if ~isfield(options,'lambda')
    options.lambda = 1.0;
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'iters')
    options.iters = 100;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
p = options.p;
lambda = options.lambda;
gamma = options.gamma;
iters = options.iters;
data = options.data;

fprintf('GCMF: data=%s  k=%d  p=%d  lambda=%f  gamma=%f\n',data,k,p,lambda,gamma);

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
Us = rand(m,k);
Ut = rand(m,k);
H = rand(k,c);
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

%% Graph Co-Regularized Nonnegative Matrix Tri-Factorization (GCMF)
Acc = [];
Obj = [];
for it = 0:iters
    
    %% Alternating Optimization
    if it>0
        Us = Us.*sqrt((Xs*(Vs*H')+lambda*Wus*Us)./(Us*(Us'*(Xs*Vs)*H')+lambda*Dus*Us+eps));
        Us = Us./(repmat(sum(Us.^2,1).^0.5,size(Us,1),1)+eps);

        Ut = Ut.*sqrt((Xt*(Vt*H')+lambda*Wut*Ut)./(Ut*(Ut'*(Xt*Vt)*H')+lambda*Dut*Ut+eps));
        Ut = Ut./(repmat(sum(Ut.^2,1).^0.5,size(Ut,1),1)+eps);

        Vs = Vs./(repmat(sum(Vs.^2,1).^0.5,size(Vs,1),1)+eps);

        Vt = Vt.*sqrt((Xt'*(Ut*H)+gamma*Wvt*Vt)./(Vt*(Vt'*(Xt'*Ut)*H)+gamma*Dvt*Vt+eps));
        Vt = Vt./(repmat(sum(Vt.^2,1).^0.5,size(Vt,1),1)+eps);
        
        H = H.*sqrt((Us'*(Xs*Vs)+Ut'*(Xt*Vt))./(Us'*(Us*H*(Vs'*Vs))+Ut'*(Ut*H*(Vt'*Vt))+eps));
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
%     O = norm(Xs-Us*(H*Vs'),'fro')^2 + norm(Xt-Ut*(H*Vt'),'fro')^2 ...
%         + lambda*trace(Us'*(Dus-Wus)*Us) + lambda*trace(Ut'*(Dut-Wut)*Ut) ...
%         + gamma*trace(Vs'*(Dvs-Wvs)*Vs) + gamma*trace(Vt'*(Dvt-Wvt)*Vt);
    Obj = [Obj;O];
    
    if mod(it,10)==0
        fprintf('[%d]  objective=%0.10f  accuracy=%0.4f\n',it,O,acc);
    end
end

fprintf('Algorithm GCMF terminated!!!\n\n\n');

end