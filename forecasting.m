function [Yhat]= forecasting(K,P,H,y,Beta)
% inputs: K is the number of variables in y_t
%         P is the lag order selection
%         H is the forecast horizon
%         y is the origin of the forecast
%         Beta - the estimated VAR coefficients
%             
% outputs: Yhat: 1- to H-step ahead prediction value for each t

A=zeros(K,K,P);
yidx=zeros(K,1,size(y,2));
yhat=zeros(K,1,size(y,2)-P,P,H);
Yhat=zeros(K,1,size(y,2)-P,H);

for t=1:1:size(y,2)
    yidx(:,:,t)=y(:,t); % index the time series 
end

% for each h, calculate the value of Ap*yt(h-p); then sum them up from
% 1 to P to get Yhat (1- to H-step ahead prediction for each time t).

for t=P:1:size(y,2)
for h=1:1:P    
for p=1:1:P
    A(:,:,p)=Beta(:,((p-1)*K+1):p*K);
    
    yhat(:,:,t,p,1)=A(:,:,p)*yidx(:,:,t-p+1); 
    Yhat(:,:,t,1)=sum(yhat(:,:,t,:,1),4); % 1-step ahead prediction
end

% for p=2:1:P
%     yhat(:,:,t,p,2)=A(:,:,p)*yidx(:,:,t-p+2);
%     for p=1:1:1
%     yhat(:,:,t,p,2)=A(:,:,p)*Yhat(:,:,t,2-p);
%     Yhat(:,:,t,2)=sum(yhat(:,:,t,:,2),4); % 2-step ahead prediction
%     end
% end
%     
% for p=3:1:P
%     yhat(:,:,t,p,3)=A(:,:,p)*yidx(:,:,t-p+3);
%     for p=1:1:2
%     yhat(:,:,t,p,3)=A(:,:,p)*Yhat(:,:,t,3-p);
%     Yhat(:,:,t,3)=sum(yhat(:,:,t,:,3),4); % 3-step ahead prediction
%     end
% end
    
for p=h:1:P
    yhat(:,:,t,p,h)=A(:,:,p)*yidx(:,:,t-p+h);
for p=1:1:h-1
    yhat(:,:,t,p,h)=A(:,:,p)*Yhat(:,:,t,h-p);
    Yhat(:,:,t,h)=sum(yhat(:,:,t,:,h),4); % h-step ahead prediction when h<=P
end
end    
end
for h=P+1:1:H
    for p=1:1:P
    yhat(:,:,t,p,h)=A(:,:,p)*Yhat(:,:,t,h-p);
    Yhat(:,:,t,h)=sum(yhat(:,:,t,:,h),4); % h-step ahead prediction when h>P
    end
end
end