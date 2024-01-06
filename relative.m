function [Stilde]=relative(K,S)
% This function is to normalize each FEVD matrix S by deviding each element
% with its respective row sum, which yields the relative connectedness
% matrix, also called as the pairwise directional connectedness.
N=size(S,3);
Stilde=zeros(size(S));
rowsum=zeros(K,1,N);
for nn=1:1:N
rowsum(:,:,nn)=sum(S(:,:,nn),2);
for i=1:1:K
for j=1:1:K
Stilde(i,j,nn)=S(i,j,nn)/rowsum(i,:,nn);
end
end
end
end