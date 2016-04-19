% Sparse probabilistic PCA

function [W,P]=sppca(t,varargin)

% Simple parser for optional variables
for narg=1:2:numel(varargin)
    eval([varargin{narg} '=varargin{' num2str(narg+1) '};']);
end

% Initialize
if ~exist('gpu','var'), gpu=false; end
if ~exist('maxiter','var'), maxiter=100; end
if ~exist('plotL','var'), plotL=false; end
if ~exist('nrestart','var'), nrestart=1; end
if ~exist('nupdate','var'), nupdate=10; end

[N,D]=size(t);
if ~exist('q','var'), q=D-1; end
a=10e-3;
b=10e-3;
c=N*D/2+a;
beta=10e-3;

if gpu, gpudev=gpuDevice(); end

% t
if gpu, t=gpuArray(t); wait(gpudev); end %#ok<*UNRCH>
sum_t2=sum(t(:).^2);

% m
um=mean(t)';

% W
disp('Performing SVD initialization of W');
[~,S,V]=svd(t,'econ');
uw=V(:,1:q)*diag(sqrt(diag(S(:,1:q))));
clear S;
if gpu
    Swi=repmat(eye(q,'gpuArray'),1,1,D);
else
    Swi=repmat(eye(q),1,1,D);
end
if gpu
    diag_uz=zeros(q,q,D,'gpuArray');
    eye_qqD=repmat(eye(q),1,1,D); %#ok<USENS>
end % optimize with sparse matrix
diag_ind=reshape(repmat(1:q+1:q^2,D,1),1,q*D)+reshape((repmat(0:q^2:q^2*(D-1),1,q)),1,q*D);
WtW=sum(Swi,3)+uw'*uw;

if plotL, h=figure; end
bestL=-inf;
for nres=1:nrestart
    
    if nrestart>1, disp(['Restart ' num2str(nres)]); end
    
    % tau
    tau=1;

    %z
    if gpu
        uz=rand(D,q,'gpuArray');
    else
        uz=rand(D,q);
    end
    
    if plotL, L=zeros(1,maxiter); end
    
    iter=0;
    prevL=-inf;
    currentL=-1e32;
    disp('Performing sparse PCA...');
    while iter<=maxiter&&currentL-prevL>thr      
        iter=iter+1;
        
        %% x
        if gpu
            Sx=pinv(eye(q,'gpuArray')+tau*WtW);
        else
            Sx=pinv(eye(q)+tau*WtW);
        end
        ux=tau*Sx*uw'*bsxfun(@minus,t,um')';
        sum_xxt=ux*ux'+N*Sx; if gpu, wait(gpudev); end
        sum_ux2=sum(diag(ux*ux'+N*Sx));
        qX=0.5*N*logdet(Sx); % -<q(X)>
        
        %% m
        if gpu
            Sm=eye(D,'gpuArray')/(beta+N*tau);
        else
            Sm=eye(D)/(beta+N*tau);
        end
        um=tau*Sm*sum(t'-uw*ux,2);
        sum_um2=sum(um.^2+diag(Sm));
        qm=0.5*logdet(Sm); % -<q(m)>
        
        %% W
        if gpu
            diag_uz(diag_ind)=uz;
            Swi=pagefun(@mtimes,diag_uz,pagefun(@inv,pagefun(@mtimes,tau*sum_xxt,diag_uz)+eye_qqD));
            uw=squeeze(tau*pagefun(@mtimes,Swi,reshape(ux*bsxfun(@minus,t,um'),q,1,D)))';
        else
            xtm=ux*bsxfun(@minus,t,um');
            for i=1:D
                Swi(:,:,i)=diag(uz(i,:))*inv(tau*sum_xxt*diag(uz(i,:))+eye(q)); %#ok<MINV>
                assert(all(eig(Swi(:,:,i))>0))
                uw(i,:)=tau*Swi(:,:,i)*xtm(:,i);
            end
        end
        WtW=sum(Swi,3)+uw'*uw;
        
        % TRY TO OPTIMIZE WITH PAGEFUN
        sum_logdet_Swi=0;
        for i=1:D
            sum_logdet_Swi=logdet(Swi(:,:,i));
        end
        qW=0.5*sum_logdet_Swi; % -<q(W)>
        
        %% z
        uz=reshape(uw(:).^2+Swi(diag_ind)',size(uw));
        qz=sum(log(uz(:))); % -<q(z)>
        
        %% tau
        d=0.5*(sum(sum(WtW.*sum_xxt))+sum_t2+N*sum_um2)+ ... % trace(<WtW><xxt>) + ||t||^2 + ||m||^2
            sum(um'*uw*ux)- ...
            sum(sum((t*uw).*ux'))- ...
            sum(t*um)+b;
        tau=c/d;
        qtau=-log(d); % -<qtauW)>
        
        %% L(q)        
        pt=-0.5*D*N*log(d)-tau*(d-b); % <p(t|theta)>
        pX=-0.5*sum_ux2; % <p(X)>
        pm=-0.5*beta*sum_um2; ... % <p(m)>
        pW=-0.5*sum(log(uz(:))); % <p(W|z)>
        pz=-sum(log(uz(:))); % <p(z)>
        ptau=-log(d); % <p(tau)>
        
        prevL=currentL;
        currentL=gather(pt+pX+pm+pW+pz+ptau+qX+qm+qW+qz+qtau);
        
        % Make sure the lower bound increases
        if iter>1
            try
                assert(prevL<=currentL);
            catch
                error('Lower bound decreased.');
            end
        end
        
        % Display lower bound evaluation
        if ~mod(iter,nupdate)
            disp(['Iter ' num2str(iter) ', L=' num2str(currentL)]);
            if plotL
                figure(h);
                L(iter)=currentL;
                plot(L(1:iter));
                pause(0.001);
            end
        end
    end
    
    % Final update of lower bound
    if plotL, figure(h); plot(L(1:iter)); end
    
    % Save output parameters
    if currentL>bestL        
        [~,ind]=sort(sum(uw.^2),'descend');
        if gpu
            W=gather(uw(:,ind));
        else
            W=uw(:,ind);
        end
        P.tau=tau;
        P.m=um;
        P.Sm=Sm;
        P.Swi=Swi;
        P.L=currentL;
        P.x=ux;
    end
end