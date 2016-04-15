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


