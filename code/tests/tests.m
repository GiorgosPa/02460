%% Simulation

N=200;

V1=randn(N,1)*sqrt(100);
V2=randn(N,1)*sqrt(98);

D=nan(N,10);
D(:,1:4)=repmat(V1,1,4)+randn(N,4);
D(:,5:8)=repmat(V2,1,4)+randn(N,4);
D(:,9:10)=randn(N,2);
% D(:,1)=D(:,1)+100;

[W,P]=vbpca(D);
W

[W,P]=sppca(D,'thr',0.001,'maxiter',500,'nrestart',1,'plotL',0,'gpu',1);
W

%% Glass data
data=importdata('glass.data');
data=data(:,2:end-1); % Remove sample index
data=bsxfun(@minus,data,mean(data));

[W,P]=vbpca(data);
W

[W,P]=sppca(data);
W

%% F data
load('Group1/group1-Fdata.mat');

[W,P]=vbpca(data);
W

[W,P]=sppca(data);
W

%% Sparse probabilistic PCA

load('Group1/group1-spData.mat');

[W,P]=vbpca(data);
W

[W,P]=sppca(data);
W

figure;
for n=1:8
    subplot(2,4,n);
    plot(W(:,n));
end

mean(W(:,1:8))
std(W(:,1:8))

%% S data
load('Group1/group1-Sdata.mat');

[W,P]=vbpca(data);
W

W=sppca(data);
W
