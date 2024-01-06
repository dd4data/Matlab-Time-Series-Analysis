function [S_out]=outofsample(T,K,Ts,y,yhat)
% T=obs-w+1-H; obs:the length of the whole time series; w: the window size;
% H: the forecast horizon.
% Ts: the size of estimation window.
% K: the dimension of yt.
% y: the time series.
% yhat: the H-step prediction.
% outputs: the ij-th element of S_out (out-of-sample approach) is the generalized variance decomposition 
%          component for a forecast error H periods ahead (KxK for each estimation point)
obs=size(y,2);
y_H=zeros(K,1,T);
e=zeros(K,1,T);
ee=zeros(K,K,T);
Sigma_out=zeros(K,K,T-Ts+1);
S_out=zeros(K,K,T-Ts+1);
num=zeros(K,K,T-Ts+1);
for m=1:1:T                               
y_H(:,:,m) =y(:,m+obs-T); % the observed value of y_t+H 
e(:,:,m)=y_H(:,:,m)-yhat(:,:,m); % forecast errors
ee(:,:,m)=e(:,:,m) * e(:,:,m)'; 
for n=1:1:T-Ts+1
    Sigma_out(:,:,n)=sum(ee(:,:,n:n+Ts-1),3)/Ts; % an estimator of sigma_out(H)
    num(:,:,n)=Sigma_out(:,:,n).^2;
for i=1:1:K
for j=1:1:K
    S_out(i,j,n)=num(i,j,n)/(Sigma_out(i,i,n)*Sigma_out(j,j,n));
end
end
end
end
end
% T=obs-w+1-H;
% y_H=zeros(K,1,obs-w+1-H);
% e=zeros(K,1,obs-w+1-H);
% ee=zeros(K,K,obs-w+1-H);
% Sigma_out=zeros(K,K,obs-H-w+1-Ts+1);
% S_out=zeros(K,K,obs-H-w+1-Ts+1);
% num=zeros(K,K,obs-H-w+1-Ts+1);

% for m=1:1:obs-H-w+1                               
% y_H(:,:,m) =y(:,m+w-1+H); % the observed value of y_t+H 
% e(:,:,m)=y_H(:,:,m)-yhat(:,:,m); % forecast errors
% ee(:,:,m)=e(:,:,m) * e(:,:,m)'; 

% for n=1:1:obs-H-w+1-Ts+1
%     Sigma_out(:,:,n)=sum(ee(:,:,n:n+Ts-1),3)/Ts; % an estimator of sigma_out(H)
%     num(:,:,n)=Sigma_out(:,:,n).^2;

% for i=1:1:K
% for j=1:1:K
%     S_out(i,j,n)=num(i,j,n)/(Sigma_out(i,i,n)*Sigma_out(j,j,n));
% end
% end
% end
% end                           