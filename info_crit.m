function [infoC, lagOrder]=info_crit(y,maxLag,inc)
% function that returns to the FPE, AIC, HQ and SC information criterion
% input: y is the observed time series data (KxT)
%        maxLag is the maximum lag order considered
% output: infoC are the values of the information criteria
%         lagOrder is the selected lag order
FPE = zeros(maxLag+1, 1);%p=0 is also considered
AIC = zeros(maxLag+1, 1);
HQ = zeros(maxLag+1, 1);
SC = zeros(maxLag+1, 1);
[K, obs]= size(y) ; 
T = obs - maxLag; % T should be same for all!
for p=0:maxLag 
    [~,~,Uhat,~]= VAR_LS(y(:,(1+(maxLag-p)):end),p,inc); 
    % ADJUST SUCH THAT Z and UHAT IS ALWAYS SAME SIZE
    sigmatilda = Uhat*Uhat'/T; 
    FPE(p+1,1)=((T+K*p+1)/(T-K*p-1))^K*det(sigmatilda);
    AIC(p+1,1)=log(det(sigmatilda))+2*K.^2*p/T;
    HQ(p+1,1)=log(det(sigmatilda))+2*K.^2*p*log(log(T))/T;
    SC(p+1,1)=log(det(sigmatilda))+K.^2*p*log(T)/T;
end
val=[FPE, AIC, SC, HQ];
[infoC,I]=min(val);
lagOrder=I-1; % for p=0 also included

