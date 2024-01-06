function [C]=netpairwise(K,Stilde)
% This function is to calculate the net pairwise directional connectedness, based on
% the pairwise directional connectedness
N=size(Stilde,3);
C=zeros(size(Stilde));
for n=1:1:N
for i=1:1:K
for j=1:1:K
C(i,j,n)=Stilde(j,i,n)- Stilde(i,j,n); % net export of from i to j.
end
end
end
end
