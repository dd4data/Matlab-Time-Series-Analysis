function [S_in]=insample(H,K,Abar,sigma,J)
% inputs: H is the forecast horizon
%         K is the number of variables in y_t
%         Abar is the coefficient matrix in the companion form of the VAR(p), KpxKp
%         J is the selection matrix (KxKp)
%         sigma is the CV matrix of the residuals
% outputs: the ij-th element of S_in (in-sample approach) is the generalized variance decomposition 
%          component for a forecast error H periods ahead (KxK)

sigma_s=diag(sigma); % the diagonal of the residual CV matrix, small sigma.
phi=zeros(K,K,H);
o=zeros(K,K,H);
u=zeros(K,K,H);
num=zeros(K,K);
den=zeros(K,K);
S_in=zeros(K,K);
for h=1:1:H
        phi(:,:,h)=J*Abar^(h-1)*J'; 
        o(:,:,h)=(phi(:,:,h)*sigma).^2;
        O(:,:)=sum(o,3);
        u(:,:,h)=phi(:,:,h)*sigma*phi(:,:,h)';
        
for j=1:1:K
        num(:,j)=O(:,j)/sigma_s(j);
        den(:,:)=sum(u,3);
        
for i=1:1:K    
S_in(i,j)=  num(i,j)/den(i,i); % ijth elment of S      

end       
end
end
end