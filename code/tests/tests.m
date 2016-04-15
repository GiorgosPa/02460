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

[W,P]=sppca(D);
W

%% Glass data
data=importdata('glass.data');
data=data(:,2:end-1); % Remove sample index
data=bsxfun(@minus,data,mean(data));

[W,P]=vbpca(data);
W

[W,P]=sppca(data);
W