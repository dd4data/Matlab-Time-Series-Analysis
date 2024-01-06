function [IC, OC, NTDC, TC]=connectedness(Stilde,K)

N=size(Stilde,3);
rowsum=zeros(K,1,N);
colsum=zeros(1,K,N);
IC=zeros(K,1,N);
OC=zeros(1,K,N);
NTDC=zeros(K,1,N);
TC=zeros(1,N); 
for n=1:1:N
rowsum(:,:,n)=sum(Stilde(:,:,n),2); % the row sum of each matrix
colsum(:,:,n)=sum(Stilde(:,:,n),1); % the column sum of each matrix
for i=1:1:K
for j=1:1:K
IC(i,:,n)=rowsum(i,:,n)-Stilde(i,i,n); % Incoming connectedness: from others
OC(:,j,n)=colsum(:,j,n)-Stilde(j,j,n); % Outgoing connectedness: to others
NTDC(i,:,n)=OC(:,i,n)-IC(i,:,n); % Net Total Directional Connectedness
end
end
TC(:,n)=sum(IC(:,:,n))/K;  % Total connectedness
end
end
