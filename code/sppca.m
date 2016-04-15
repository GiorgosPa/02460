% Sparse probabilistic PCA

function [W,P]=sppca(t)

maxiter=100;

% Initialize
[N,D]=size(t);
q=D-1;

% Initialize W to PCA solution
disp('Performing SVD');
[~,S,V]=svd(t,'econ');
uw=V(:,1:q)*diag(sqrt(diag(S(:,1:q))));
% uw=randn(D,q);

Swi=repmat(eye(q),1,1,D);
WtW=sum(Swi,3)+uw'*uw;

a=10e-3;
b=10e-3;
c=N*D/2+a;
tau=1;
beta=10e-3;

um=mean(t)';
uz=rand(D,q);

sum_t2=sum(t(:).^2);

L=zeros(1,maxiter);
pt=zeros(1,maxiter); pX=zeros(1,maxiter); pm=zeros(1,maxiter); pW=zeros(1,maxiter); pz=zeros(1,maxiter);
ptau=zeros(1,maxiter); qX=zeros(1,maxiter); qm=zeros(1,maxiter); qW=zeros(1,maxiter);
qz=zeros(1,maxiter); qtau=zeros(1,maxiter);
d_tmp=zeros(1,maxiter);

h=figure;
iter=0;
disp('Performing sparse PCA...');
while iter<=maxiter
    iter=iter+1;
    
    if ~mod(iter,10)
        disp(['Iter ' num2str(iter)]);
        figure(h);
        plot(L(1:iter));
        pause(0.001);
    end
    
    %% x
    Sx=pinv(eye(q)+tau*WtW);
    ux=tau*Sx*uw'*bsxfun(@minus,t,um')';
    sum_xxt=ux*ux'+N*Sx;
    qX(iter)=0.5*N*logdet(Sx); % -<q(X)>
    
    %% m
    Sm=eye(D)/(beta+N*tau);
    um=tau*Sm*sum(t'-uw*ux,2);
    sum_um2=sum(um.^2+diag(Sm));
    qm(iter)=0.5*logdet(Sm); % -<q(m)>
        
    %% W
    tnk=bsxfun(@minus,t,um');
    for i=1:D
        Swi(:,:,i)=diag(uz(i,:))*inv(tau*sum_xxt*diag(uz(i,:))+eye(q)); %#ok<MINV>
        assert(all(eig(Swi(:,:,i))>0))
        xtm=sum(bsxfun(@times,ux,tnk(:,i)'),2);
        uw(i,:)=tau*Swi(:,:,i)*xtm;
    end
    WtW=sum(Swi,3)+uw'*uw;
    
    sum_logdet_Swi=0;
    for i=1:D
        sum_logdet_Swi=logdet(Swi(:,:,i));
    end
    qW(iter)=0.5*sum_logdet_Swi; % -<q(W)>
    
    %% z
    for i=1:D
        uz(i,:)=uw(i,:).^2+diag(Swi(:,:,i))';
    end
    qz(iter)=sum(log(uz(:))); % -<q(z)>
    
    %% tau   
    d=0.5*(sum(sum(WtW.*sum_xxt))+sum_t2+N*sum_um2)+ ... % trace(<WtW><xxt>) + ||t||^2 + ||m||^2
        sum(um'*uw*ux)- ...
        sum(sum((t*uw).*ux'))- ...
        sum(t*um)+b;
    d_tmp(iter)=d;
    
    tau=c/d;
    qtau(iter)=-log(d); % -<qtauW)>    
        
    %% L(q)
    
    sum_ux2=sum(diag(ux*ux'+N*Sx));
    pt(iter)=-0.5*D*N*log(d)-tau*(d-b); % <p(t|theta)>
    pX(iter)=-0.5*sum_ux2; % <p(X)>
    pm(iter)=-0.5*beta*sum_um2; ... % <p(m)>
    pW(iter)=-0.5*sum(log(uz(:))); % <p(W|z)>
    pz(iter)=-sum(log(uz(:))); % <p(z)>
    ptau(iter)=-log(d); % <p(tau)>

    L(iter)=pt(iter)+pX(iter)+pm(iter)+pW(iter)+pz(iter)+ptau(iter)+...
        qX(iter)+qm(iter)+qW(iter)+qz(iter)+qtau(iter);
    
    
    % Make sure the lower bound increases
    if iter>1
        try
            assert(L(iter)>=L(iter-1));
        catch
            error('Lower bound increased.');
        end
    end
end

% Reorder W according to largest PC
[~,ind]=sort(sum(uw.^2),'descend');
W=uw(:,ind);

% Save output parameters
P.tau=tau;
P.m=um;
P.Sm=Sm;
P.Swi=Swi;
P.L=L;
P.x=ux;

% Final update of lower bound
figure(h); plot(L(1:iter));