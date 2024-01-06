function [lambda_BP, p_boot_BP, p_chi_BP, lambda_SS, p_boot_SS, p_chi_SS]= chowtests(y,Tb,p,inc,R)
%inputs:  y is the observed data (KxT);
%         Tb is the break date;
%         p is the lag order of VAR;
%         inc indicates there is an intercept when inc=1;
%         R is the amount of replication for bootstrap.
% outputs:lambda_BP is the Chow break point test statistics;
%         p_boot_BP is the p-value of Chow break point test based on the
%         bootstrap;
%         p_chi_BP is the p-value of Chow break point test based on the
%         chi-squared distribution;
%         lambda_SS is the Chow split sample test statistics;
%         p_boot_SS is the p-value of Chow sample split test based on the
%         bootstrap;
%         p_chi_SS is the p-value of Chow sample split test based on the 
%         chi-squared distribution;

[K, obs]=size(y);  % obs:number of total observations
[Bhat_12,~,Uhat_12,~]= VAR_LS(y,p,inc); 
[~,~,Uhat_1,~]= VAR_LS(y(:,1:Tb-1),p,inc); 
[~,~,Uhat_2,~]= VAR_LS(y(:,Tb:end),p,inc); 
T1=length(Uhat_1);
T2=length(Uhat_2);
T=length(Uhat_12);
sigmatilda_1 = Uhat_1*Uhat_1'/T1;
sigmatilda_2 = Uhat_2*Uhat_2'/T2;
sigmatilda_12= 1./(T1+T2)*( Uhat_12(:,1:T1)*Uhat_12(:,1:T1)' + Uhat_12(:,(end-T2+1):end)*Uhat_12(:,(end-T2+1):end)' ) ; 
                                         
lambda_BP=(T1+T2)*log(det(sigmatilda_12))-T1*log(det(sigmatilda_1))-T2*log(det(sigmatilda_2));
lambda_SS=(T1+T2)*(log(det(sigmatilda_12))-log(det((T1+T2).^(-1)*(T1*sigmatilda_1+T2*sigmatilda_2))));
p_chi_BP=1-chi2cdf(lambda_BP,(K*(K*p+1)+K*(K+1)/2));
p_chi_SS=1-chi2cdf(lambda_SS,(K*(K*p+1)));

% Bootstrap:
Ucen=zeros(K,T);
for t=1:obs-p
Ucen(:,t)=Uhat_12(:,t)-mean(Uhat_12,2); % centered residuals Ucen

end
% The following code constructs a  bootstrap time series, 
% based on your estimated VAR(p) residuals "ustar":
burnin=50;
v=Bhat_12(:,1);
ytemp = zeros(K,obs+burnin); % One Bootstrap Time Series

B_BP=zeros(R,1);
B_SS=zeros(R,1);
Nb_BP=zeros(R,1);
Nb_SS=zeros(R,1);
for bs=1:1:R
ubs=Ucen(:,randi([1 T],1,obs+burnin)); % bootstrap residuals resampled from fitted residuals with replacement
ytemp(:,1:p)= v*ones(1,p) + ubs(:,1:p); 
for t = p+1:(obs+burnin) % This Loop constructs VAR data based on residuals "ubs"
ytemp(:,t)= v + Bhat_12(:,2:end)* reshape(ytemp(:,t-1:-1:t-p),p*K,1) + ubs(:,t);
end
ybs = ytemp(:,burnin+1:end); % Kick out first "burnin" observations, the rest is your "final" bootstrap time series

[Bstar,~,Ustar,~]= VAR_LS(ybs,p,inc); % re-estimation using the bootstrap time series

[~,~,Ustar_1,~]= VAR_LS(ybs(:,1:Tb-1),p,inc); 
[~,~,Ustar_2,~]= VAR_LS(ybs(:,Tb:end),p,inc); 
Tstar1=length(Ustar_1);
Tstar2=length(Ustar_2);
Tstar=length(Ustar);
sigmastar_1 = Ustar_1*Ustar_1'/Tstar1;
sigmastar_2 = Ustar_2*Ustar_2'/Tstar2;
sigmastar= 1./(Tstar1+Tstar2)*( Ustar(:,1:Tstar1)*Ustar(:,1:Tstar1)' + Ustar(:,(end-Tstar2+1):end)*Ustar(:,(end-Tstar2+1):end)' ) ; 
                                         
lamstar_BP=(Tstar1+Tstar2)*log(det(sigmastar))-Tstar1*log(det(sigmastar_1))-Tstar2*log(det(sigmastar_2));
lamstar_SS=(Tstar1+Tstar2)*(log(det(sigmastar))-log(det((Tstar1+Tstar2).^(-1)*(Tstar1*sigmastar_1+Tstar2*sigmastar_2))));
    
B_BP(bs,:)=lamstar_BP;
B_SS(bs,:)=lamstar_SS;

if B_BP(bs,:)>lambda_BP
Nb_BP(bs,:)=1;
else
Nb_BP(bs,:)=0;
end

if B_SS(bs,:)>lambda_SS
    Nb_SS(bs,:)=1;
else
    Nb_SS(bs,:)=0;
end
end
p_boot_BP=sum(Nb_BP,1)/R;
p_boot_SS=sum(Nb_SS,1)/R;
end