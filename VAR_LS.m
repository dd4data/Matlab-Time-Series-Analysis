function [Bhat,sigmahat,Uhat,Tstat]= VAR_LS(y,p,inc)
% inputs: y is the observed data
%         p is the lag order of VAR
%         inc indicates there is an intercept when inc=1
% outputs: Bhat are the least square estimates of the VAR parameters
%          sigmahat is the least square estimate of the residual CV matrix
%          Uhat are the residuals
%          Tstat are t-statistics of estimated parameters
[K, obs]=size(y);  % number of total observations
T=obs-p;

Y=y(:,p+1:end);
lax=lagmatrix(y',1:p);
Z=lax(p+1:end,:)';

if inc==1
    Z=[ones(1,T);Z];
end
Bhat=Y*Z'*inv(Z*Z'); %LS Estimator
Uhat=(Y-Bhat*Z);
sigmahat=1/(T-K*p-1)*Uhat*Uhat'; % Residual Cov-Matrix
VarBhat=kron(inv(Z*Z'),sigmahat); % Cov Matrix of Estimated Parameters
se=diag(VarBhat).^0.5; % Standard Errors of the Estimated Parameters
Tstat=reshape((Bhat(:)./se),K,K*p+inc); % T statistics of the Estimated Parameters

