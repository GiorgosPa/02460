% Variational bayese PCA

function [W,P]=vbpca(t)

maxiter=100;

[N,d]=size(t);
q=d-1;

% Initialize
[~,S,V]=svd(t);
mw=V(:,1:q)*diag(sqrt(diag(S(:,1:q))));
% mw=randn(d,q);
Sw=eye(q);
WtW=Sw+mw'*mw;

mu=mean(t)';
sum_t2=sum(t(:).^2);

aa0=10e-3;
aa=aa0+d/2;
ba0=ones(1,q)*10e-3;
at0=10e-3;
at=at0+N*d/2;
bt0=10e-3;
alpha=ones(1,q);
tau=1;
beta=10e-3;

iter=1;
L=nan(1,maxiter);
while iter<maxiter
    if ~mod(iter,50), disp(['Iter ' num2str(iter)]); end
    
    %% X
    Sx=inv(eye(q)+tau*WtW);
    mx=tau*Sx*mw'*bsxfun(@minus,t,mu')'; %#ok<*MINV>
    sum_xxt=mx*mx'+N*Sx;
    
    %% u
    Su=inv(beta+N*tau)*eye(d);
    mu=tau*Su*sum(t'-mw*mx,2);
    sum_mu2=sum(mu.^2+diag(Su));
    
    %% W
    Sw=inv(diag(alpha)+tau*sum_xxt);    
    tnk=bsxfun(@minus,t,mu');
    for i=1:d
        xtm=sum(bsxfun(@times,mx,tnk(:,i)'),2);
        mw(i,:)=tau*Sw*xtm;
    end
    WtW=d*Sw+mw'*mw;
    
    %% Alpha
    ba=ba0+sum(diag(WtW))/2;
    alpha=aa./ba;
    
    %% Tau
    bt=0.5*(sum(sum(WtW.*sum_xxt))+sum_t2+N*sum_mu2)+ ... % trace(<WtW><xxt>) + ||t||^2 + ||m||^2
        sum(mu'*mw*mx)- ...
        sum(sum((t*mw).*mx'))- ...
        sum(t*mu)+bt0;
    tau=at/bt;
        
    %% Lower bound
    
    sum_mx2=sum(diag(mx*mx'+N*Sx));   
    L(iter)=-0.5*d*N*log(bt)-tau*(bt-bt0) ... % <p(t|theta)>
        - 0.5*sum_mx2 ... % <p(x)>
        - 0.5*beta*sum_mu2 ... % <p(m)>
        - 0.5*(d*sum(log(ba))+sum(alpha.*diag(WtW)')) ... % <p(W|alpha)>
        + sum(log(ba))	 ... % <p(alpha)>
        + log(bt) ... % <p(tau)>
        + 0.5*N*logdet(Sx) ... % -<q(X)>
        + 0.5*logdet(Su) ... % -<q(m)>
        + 0.5*d*logdet(Sw) ... % -<q(W)>
        - sum(log(ba)) ... % -<q(alpha)>
        - log(bt); % -<q(tau)>
    
    assert(~isinf(L(iter)));
    % Make sure the lower bound increases
    if iter>1
        try
            assert(L(iter)-L(iter-1)>=-1);
        catch
            error('Lower bound increased.');
        end
    end
    
    iter=iter+1;
end

figure(1); plot(L);

[~,ind]=sort(sum(mw.^2),2,'descend');
W=mw(:,ind);
P.alpha=alpha(ind);
P.L=L;
P.u=mu;