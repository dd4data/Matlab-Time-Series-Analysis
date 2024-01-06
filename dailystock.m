%% import and process data
clear;clc;

data=readmatrix('daily stock 06-19.xlsx');
SNP=data(:,1:2); FTSE=data(:,3:4); DAX=data(:,5:6); CAC=data(:,7:8);
MIB=data(:,9:10); NIKKEI=data(:,11:12); HS=data(:,13:14); SH=data(:,15:16);

snpdates=datetime(SNP(:,1),'ConvertFrom','excel'); % convert the date from excel
snpdates=datenum(snpdates); % convert the datestr to datenum
SNP=[snpdates, SNP(:,2)];  % change the date into matlab datenum

ftsedates=datetime(FTSE(:,1),'ConvertFrom','excel');
ftsedates=datenum(ftsedates); 
FTSE=[ftsedates, FTSE(:,2)];

daxdates=datetime(DAX(:,1),'ConvertFrom','excel');
daxdates=datenum(daxdates); 
DAX=[daxdates, DAX(:,2)];

cacdates=datetime(CAC(:,1),'ConvertFrom','excel');
cacdates=datenum(cacdates); 
CAC =[cacdates, CAC(:,2)];

mibdates=datetime(MIB(:,1),'ConvertFrom','excel');
mibdates=datenum(mibdates); 
MIB =[mibdates, MIB(:,2)];

nidates=datetime(NIKKEI(:,1),'ConvertFrom','excel');
nidates=datenum(nidates); 
NIKKEI =[nidates, NIKKEI(:,2)];

hsdates=datetime(HS(:,1),'ConvertFrom','excel');
hsdates=datenum(hsdates); 
HS =[hsdates, HS(:,2)];

shdates=datetime(SH(:,1),'ConvertFrom','excel');
shdates=datenum(shdates); 
SH =[shdates, SH(:,2)];

bdates = busdays('01/01/2006','05/31/2019');
daily=zeros(size(bdates,1), 9);

for i=1:size(bdates,1)
    daily(i,1)=bdates(i,1);
    
for j=1:size(SNP,1)
if  daily(i,1)==SNP(j,1)
       daily(i,2)=SNP(j,2); 
% else 
%     friday(i,2)=NaN;
end
end
end

for i=1:size(bdates,1)
for k=1:size(FTSE,1)
if  daily(i,1)==FTSE(k,1)
       daily(i,3)=FTSE(k,2);

end
end
end

for i=1:size(bdates,1)
for l=1:size(DAX,1)
if  daily(i,1)==DAX(l,1)
       daily(i,4)=DAX(l,2); 

end
end
end

for i=1:size(bdates,1)
for m=1:size(CAC,1)
if  daily(i,1)==CAC(m,1)
       daily(i,5)=CAC(m,2); 

end
end
end

for i=1:size(bdates,1)
for n=1:size(MIB,1)
if  daily(i,1)==MIB(n,1)
       daily(i,6)=MIB(n,2); 

end
end
end

for i=1:size(bdates,1)
for o=1:size(NIKKEI,1)
if  daily(i,1)==NIKKEI(o,1)
       daily(i,7)=NIKKEI(o,2);

end
end
end

for i=1:size(bdates,1)
for q=1:size(HS,1)
if  daily(i,1)==HS(q,1)
       daily(i,8)=HS(q,2);

end
end
end 

for i=1:size(bdates,1)
for r=1:size(SH,1)
if  daily(i,1)==SH(r,1)
       daily(i,9)=SH(r,2); 

end
end
end

indice=daily(2:end,2:end); % stock market indices are ready, from Jan 4th, 2006 to May 31th, 2019!!!
                           % holidays followed from USA.
for i=1:size(indice,1)
    for j=1:size(indice,2)
    if indice(i,j)==0
        indice(i,j)=NaN;  % to find missing data.
    end
        
    end
end

indice=fillmissing(indice,'previous',1); % fill missing data with previous ones.

lin=log(indice);                          
dreturn=diff(lin,1,1); % daily return of the stock markets:from Jan 5th, 2006 to May 31th, 2019
y=dreturn'; % transform observations into KxT dimension.

%% data description
[K,obs]=size(y); 

% x=1:size(y,2);
% figure(1)
% plot(x,y(1,:),x,y(2,:),x,y(3,:),x,y(4,:),x,y(5,:),x,y(6,:),...
%     x,y(7,:),x,y(8,:)); % plot of the daily return of the stock markets


formatOut = 'dd/mm/yyyy'; Batestr=datestr(bdates,formatOut); % plot dates
B = size(Batestr,1);
b =337.5;
figure(1) % plot of the daily return of the stock markets
plot(y(1,:),'color','[0 0.4470 0.7410]','DisplayName', 'USA');
hold on;
plot(y(2,:),'r','DisplayName', 'UK');
hold on;
plot(y(3,:),'k','DisplayName', 'GER');
hold on;
plot(y(4,:),'b','DisplayName', 'FRA');
hold on;
plot(y(5,:),'g','DisplayName', 'IT');
hold on;
plot(y(6,:),'color','[0.9290 0.6940 0.1250]','DisplayName', 'JPN');
hold on;
plot(y(7,:),'color','[0.4940 0.1840 0.5560]','DisplayName', 'HK');
hold on;
plot(y(8,:),'color','[0.6350 0.0780 0.1840]','DisplayName', 'CHN');
h1 = gca;
h1.XLim = [0,B];
h1.XTick = 1:b:B;
h1.XTickLabel = Batestr(1:b:B,4:end);

[h1,pValue1,stat1,cValue1,~] =adftest(y(1,:));  % ADF-test
[h2,pValue2,stat2,cValue2,~] =adftest(y(2,:)); 
[h3,pValue3,stat3,cValue3,~] =adftest(y(3,:)); 
[h4,pValue4,stat4,cValue4,~] =adftest(y(4,:)); 
[h5,pValue5,stat5,cValue5,~] =adftest(y(5,:)); 
[h6,pValue6,stat6,cValue6,~] =adftest(y(6,:)); 
[h7,pValue7,stat7,cValue7,~] =adftest(y(7,:)); 
[h8,pValue8,stat8,cValue8,~] =adftest(y(8,:)); 
 
adf=[h1,pValue1,stat1,cValue1;
     h2,pValue2,stat2,cValue2;
     h3,pValue3,stat3,cValue3;
     h4,pValue4,stat4,cValue4; 
     h5,pValue5,stat5,cValue5; 
     h6,pValue6,stat6,cValue6; 
     h7,pValue7,stat7,cValue7; 
     h8,pValue8,stat8,cValue8];
table_adf=array2table(adf,...
    'VariableNames', {'test_decision', 'p_value', 'test_statistics', 'critical_value'}, ...
    'RowNames', {'USA','UK','GERMANY','FRANCE','ITALY','JAPAN','HONGKONG','CHINA'});
 
disp(table_adf);
%h = 1 indicates rejection of the unit-root null in favor of the alternative model.
%h = 0 indicates failure to reject the unit-root null. 

% description of the data
M = mean(y,2);
Md= median(y,2);
Max= max(y,[],2);
Min=min(y,[],2);
desp=round([M, Md, Max, Min],4);
table_desp=array2table(desp,...
    'VariableNames', {'Mean', 'Median', 'Maximum', 'Minimum'}, ...
    'RowNames', {'USA','UK','GERMANY','FRANCE','ITALY','JAPAN','HONGKONG','CHINA'});
disp(table_desp);

%% Unconditional:
maxLag=8; 
inc=0;
% H=1;
[infoC, lagOrder]=info_crit(y,maxLag,inc);
p=lagOrder(:,3); 
[Bhat,sigmahat,Uhat,Tstat]= VAR_LS(y,p,inc); % estimate a VAR(p)
[~, Abar, J]=tr2VAR1(y,Bhat,p,inc); % transfer VAR(p) into VAR(1)
S1_full=insample(1,K,Abar,sigmahat,J);  % compute s_ij for in-sample FEVD for the full sample
S2_full=insample(2,K,Abar,sigmahat,J);  % compute s_ij for in-sample FEVD for the full sample
S5_full=insample(5,K,Abar,sigmahat,J);  % compute s_ij for in-sample FEVD for the full sample

Stilde1_full=relative(K,S1_full);       % calculate the relative pairwise connectedness matrix
Stilde2_full=relative(K,S2_full);       % calculate the relative pairwise connectedness matrix
Stilde5_full=relative(K,S5_full);       % calculate the relative pairwise connectedness matrix
Stilde_full=(Stilde1_full+Stilde2_full+Stilde5_full)/3; % take the average

names = {'USA' 'UK' 'GER' 'FRA' 'IT' 'JPN' 'HK' 'CHN'};

figure(2) % full-sample network connectedness
G = digraph(Stilde_full',names,'omitselfloops');
LWidths = G.Edges.Weight*10;
plot(G,'LineWidth',LWidths,'EdgeColor','r','NodeColor','k','Layout','circle');




[IC_full, OC_full, NTDC_full, TC_full]=connectedness(Stilde_full,K);  % calculate the connectedness
Full=round([Stilde_full, IC_full; OC_full, TC_full]*100,2);
table_Full=array2table(Full,...
    'VariableNames', {'USA','UK','GERMANY','FRANCE','ITALY','JAPAN','HONGKONG','CHINA','Incoming'}, ...
    'RowNames', {'USA','UK','GERMANY','FRANCE','ITALY','JAPAN','HONGKONG','CHINA','Outgoing'});
disp(table_Full);
%% Conditional(rolling-sample approach):
maxLag=8; 
inc=0;
w=250; % set the window size
Ts=250; % set the estimation window
N=size(y,2)-w+1; % # of samples equates the length of the time series minus the window size, then plus one.

S_in1=zeros(K,K,N);S_in2=zeros(K,K,N);S_in5=zeros(K,K,N);
yhat=zeros(K,1,N);

Lfew=zeros(N,4);
for nn=1:1:N 
    Y=y(:,nn:nn+w-1); % sampling for each window

    [IC, LO]=info_crit(Y,maxLag,inc); % selecting lag-order for each window
                                      %IC:information criteria; LO:lag-order.
    %P=LO(:,2); % suppose using suggested lag order by SC
    
    Lfew(nn,:)=LO;  % Lag-order for each window
end


%% Compute different results with different horizons and take the average
H=1; % H is the forecast horizon
T=obs-w+1-H;
 for nn=1:1:N 
    Y=y(:,nn:nn+w-1); % sampling for each window
     P=1;
    [Beta,sigma,U,~]= VAR_LS(Y,P,inc); % LS-estimation for each window
    
    [vbar, Abar, J]=tr2VAR1(Y,Beta,P,inc); % transfer VAR(p) into VAR(1)
   
[S_in1(:,:,nn)]=insample(H,K,Abar,sigma,J);  % compute s_ij for in-sample FEVD

[Yhat]= forecasting(K,P,H,Y,Beta); % forecasting with each t in the window as an origin

yhat(:,:,nn)=Yhat(:,:,end,end); % for each window,take out the H-step forecast of the last origin;
                               % yields yhat, which are used to compute
                               % sigma_out(H),where t=w:end.

end
[S_out1]=outofsample(T,K,Ts,y,yhat); % compute s_ij for out-of-sample FEVD


H=2; % H is the forecast horizon
T=obs-w+1-H;
 for nn=1:1:N 
    Y=y(:,nn:nn+w-1); % sampling for each window
     P=1;
    [Beta,sigma,U,~]= VAR_LS(Y,P,inc); % LS-estimation for each window
    
    [vbar, Abar, J]=tr2VAR1(Y,Beta,P,inc); % transfer VAR(p) into VAR(1)
   
[S_in2(:,:,nn)]=insample(H,K,Abar,sigma,J);  % compute s_ij for in-sample FEVD

[Yhat]= forecasting(K,P,H,Y,Beta); % forecasting with each t in the window as an origin

yhat(:,:,nn)=Yhat(:,:,end,end); % for each window,take out the H-step forecast of the last origin;
                               % yields yhat, which are used to compute
                               % sigma_out(H),where t=w:end.

end
[S_out2]=outofsample(T,K,Ts,y,yhat); % compute s_ij for out-of-sample FEVD



H=5; % H is the forecast horizon
T=obs-w+1-H;
 for nn=1:1:N 
    Y=y(:,nn:nn+w-1); % sampling for each window
     P=1;
    [Beta,sigma,U,~]= VAR_LS(Y,P,inc); % LS-estimation for each window
    
    [vbar, Abar, J]=tr2VAR1(Y,Beta,P,inc); % transfer VAR(p) into VAR(1)
   
[S_in5(:,:,nn)]=insample(H,K,Abar,sigma,J);  % compute s_ij for in-sample FEVD

[Yhat]= forecasting(K,P,H,Y,Beta); % forecasting with each t in the window as an origin

yhat(:,:,nn)=Yhat(:,:,end,end); % for each window,take out the H-step forecast of the last origin;
                               % yields yhat, which are used to compute
                               % sigma_out(H),where t=w:end.

end
[S_out5]=outofsample(T,K,Ts,y,yhat); % compute s_ij for out-of-sample FEVD

S_in=(S_in1+S_in2+S_in5)/3;  % to take the average of FEVD of forecast horizons=1,2,5.
S_out1=S_out1(:,:,5:end); S_out2=S_out2(:,:,4:end);
S_out=(S_out1+S_out2+S_out5)/3;

[Stilde_in]=relative(K,S_in);       % calculate the relative connectedness matrix
[Stilde_out]=relative(K,S_out);     % also called the pairwise directional connectedness

[C_in]=netpairwise(K,Stilde_in);
[C_out]=netpairwise(K,Stilde_out);  % calculate the net pairwise directional connectedness

[IC_in, OC_in, NTDC_in, TC_in]=connectedness(Stilde_in,K);

[IC_out, OC_out, NTDC_out, TC_out]=connectedness(Stilde_out,K); % calculate total directional connectedness,
                                    % net total directional connectedness
                                    % and total connectedness.
%C_in= C_in(:,:,Ts+H:end); % to take the same length of the out-of-sample approach.

TC_in_s=TC_in(:,Ts+H:end); % to take the same length as the out-of-sample approach

OC_in=OC_in(:,:,Ts+H:end); % to take the same length of the out-of-sample approach.

IC_in=IC_in(:,:,Ts+H:end); % to take the same length of the out-of-sample approach.

NTDC_in=NTDC_in(:,:,Ts+H:end); % to take the same length of the out-of-sample approach.

diff=abs(OC_out-OC_in)./OC_in; % compute the relative difference of country-wise Out- and In- relative connectedness
aggdiff=sum(diff,2); % compute the aggregated difference
AD=zeros(1,size(aggdiff,3));
for t=1:1:size(aggdiff,3)
    AD(:,t)=aggdiff(:,:,t); % transfer the dimension in order to plot the aggregated difference
end

% e1=zeros(K,size(OC_out,3));
% e2=zeros(K,size(OC_in,3));
% DM=zeros(K,1);             
% for kk=1:1:size(OC_out,2)  
%     cc=1:1:size(OC_out,3);
% e1(kk,cc)=OC_out(:,kk,cc); 
% e2(kk,cc)=OC_in(:,kk,cc);     
% 
% end
% for kk=1:1:size(OC_out,2)
% DM(kk,:) = dmtest(e1(kk,:)', e2(kk,:)', H);
% % to perform the Diebold-Mariano Test with H0: OC_out=OC_in.
% end
% pDM_oc =pdf('normal',DM,0,1); % p-value of the DM test on Outgoing connectedness
% display(pDM_oc);
% 
% dmstat = dmtest(TC_out', TC_in_s', H);% p-value of the DM test on Total connectedness
% pdm_tc=pdf('normal',dmstat,0,1);

oc_in_usa=zeros(1,size(OC_in,3));
oc_out_usa=zeros(1,size(OC_out,3));
ic_in_usa=zeros(1,size(IC_in,3));
ic_out_usa=zeros(1,size(IC_out,3));
ntdc_in_usa=zeros(1,size(NTDC_in,3));
ntdc_out_usa=zeros(1,size(NTDC_out,3));
diff_usa=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_usa(:,t)=OC_in(:,1,t);
oc_out_usa(:,t)=OC_out(:,1,t);
ic_in_usa(:,t)=IC_in(1,:,t);
ic_out_usa(:,t)=IC_out(1,:,t);
ntdc_in_usa(:,t)=NTDC_in(1,:,t);
ntdc_out_usa(:,t)=NTDC_out(1,:,t);
diff_usa(:,t)=diff(:,1,t);
end

oc_in_uk=zeros(1,size(OC_in,3));
oc_out_uk=zeros(1,size(OC_out,3));
ic_in_uk=zeros(1,size(IC_in,3));
ic_out_uk=zeros(1,size(IC_out,3));
ntdc_in_uk=zeros(1,size(NTDC_in,3));
ntdc_out_uk=zeros(1,size(NTDC_out,3));
diff_uk=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_uk(:,t)=OC_in(:,2,t);
oc_out_uk(:,t)=OC_out(:,2,t);
ic_in_uk(:,t)=IC_in(2,:,t);
ic_out_uk(:,t)=IC_out(2,:,t);
ntdc_in_uk(:,t)=NTDC_in(2,:,t);
ntdc_out_uk(:,t)=NTDC_out(2,:,t);
diff_uk(:,t)=diff(:,2,t);
end

oc_in_deu=zeros(1,size(OC_in,3));
oc_out_deu=zeros(1,size(OC_out,3));
ic_in_deu=zeros(1,size(IC_in,3));
ic_out_deu=zeros(1,size(IC_out,3));
ntdc_in_deu=zeros(1,size(NTDC_in,3));
ntdc_out_deu=zeros(1,size(NTDC_out,3));
diff_deu=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_deu(:,t)=OC_in(:,3,t);
oc_out_deu(:,t)=OC_out(:,3,t);
ic_in_deu(:,t)=IC_in(3,:,t);
ic_out_deu(:,t)=IC_out(3,:,t);
ntdc_in_deu(:,t)=NTDC_in(3,:,t);
ntdc_out_deu(:,t)=NTDC_out(3,:,t);
diff_deu(:,t)=diff(:,3,t);
end

oc_in_fra=zeros(1,size(OC_in,3));
oc_out_fra=zeros(1,size(OC_out,3));
ic_in_fra=zeros(1,size(IC_in,3));
ic_out_fra=zeros(1,size(IC_out,3));
ntdc_in_fra=zeros(1,size(NTDC_in,3));
ntdc_out_fra=zeros(1,size(NTDC_out,3));
diff_fra=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_fra(:,t)=OC_in(:,4,t);
oc_out_fra(:,t)=OC_out(:,4,t);
ic_in_fra(:,t)=IC_in(4,:,t);
ic_out_fra(:,t)=IC_out(4,:,t);
ntdc_in_fra(:,t)=NTDC_in(4,:,t);
ntdc_out_fra(:,t)=NTDC_out(4,:,t);
diff_fra(:,t)=diff(:,4,t);
end

oc_in_itly=zeros(1,size(OC_in,3));
oc_out_itly=zeros(1,size(OC_out,3));
ic_in_itly=zeros(1,size(IC_in,3));
ic_out_itly=zeros(1,size(IC_out,3));
ntdc_in_itly=zeros(1,size(NTDC_in,3));
ntdc_out_itly=zeros(1,size(NTDC_out,3));
diff_itly=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_itly(:,t)=OC_in(:,5,t);
oc_out_itly(:,t)=OC_out(:,5,t);
ic_in_itly(:,t)=IC_in(5,:,t);
ic_out_itly(:,t)=IC_out(5,:,t);
ntdc_in_itly(:,t)=NTDC_in(5,:,t);
ntdc_out_itly(:,t)=NTDC_out(5,:,t);
diff_itly(:,t)=diff(:,5,t);
end

oc_in_jpn=zeros(1,size(OC_in,3));
oc_out_jpn=zeros(1,size(OC_out,3));
ic_in_jpn=zeros(1,size(IC_in,3));
ic_out_jpn=zeros(1,size(IC_out,3));
ntdc_in_jpn=zeros(1,size(NTDC_in,3));
ntdc_out_jpn=zeros(1,size(NTDC_out,3));
diff_jpn=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_jpn(:,t)=OC_in(:,6,t);
oc_out_jpn(:,t)=OC_out(:,6,t);
ic_in_jpn(:,t)=IC_in(6,:,t);
ic_out_jpn(:,t)=IC_out(6,:,t);
ntdc_in_jpn(:,t)=NTDC_in(6,:,t);
ntdc_out_jpn(:,t)=NTDC_out(6,:,t);
diff_jpn(:,t)=diff(:,6,t);
end

oc_in_hk=zeros(1,size(OC_in,3));
oc_out_hk=zeros(1,size(OC_out,3));
ic_in_hk=zeros(1,size(IC_in,3));
ic_out_hk=zeros(1,size(IC_out,3));
ntdc_in_hk=zeros(1,size(NTDC_in,3));
ntdc_out_hk=zeros(1,size(NTDC_out,3));
diff_hk=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_hk(:,t)=OC_in(:,7,t);
oc_out_hk(:,t)=OC_out(:,7,t);
ic_in_hk(:,t)=IC_in(7,:,t);
ic_out_hk(:,t)=IC_out(7,:,t);
ntdc_in_hk(:,t)=NTDC_in(7,:,t);
ntdc_out_hk(:,t)=NTDC_out(7,:,t);
diff_hk(:,t)=diff(:,7,t);
end

oc_in_chn=zeros(1,size(OC_in,3));
oc_out_chn=zeros(1,size(OC_out,3));
ic_in_chn=zeros(1,size(IC_in,3));
ic_out_chn=zeros(1,size(IC_out,3));
ntdc_in_chn=zeros(1,size(NTDC_in,3));
ntdc_out_chn=zeros(1,size(NTDC_out,3));
diff_chn=zeros(1,size(diff,3));
for t=1:1:size(OC_in,3)
oc_in_chn(:,t)=OC_in(:,8,t);
oc_out_chn(:,t)=OC_out(:,8,t);
ic_in_chn(:,t)=IC_in(8,:,t);
ic_out_chn(:,t)=IC_out(8,:,t);
ntdc_in_chn(:,t)=NTDC_in(8,:,t);
ntdc_out_chn(:,t)=NTDC_out(8,:,t);
diff_chn(:,t)=diff(:,8,t);
end

bpdates = busdays('01/01/2006','05/31/2019'); % to plot the date evenly
% Date=bdates(w+Ts+H+1:end,:); 
formatOut = 'dd/mm/yy'; Datestr=datestr(bpdates(w+Ts+H+1:end,:),formatOut); % count the respective dates
D = size(Datestr,1);
d =205;

%% Networks: difference of pairwise directional connectedness btw. in-sample and out-of-sample
Cout=Stilde_out;
Cin=Stilde_in(:,:,Ts+H:end);
net=Cout-Cin; % difference of pairwise directional connectedness btw. in-sample and out-of-sample
Q1=zeros(K,K);Q2=zeros(K,K);Q3=zeros(K,K);
for i=1:1:size(net,1)
for j=1:1:size(net,2)
Q1(i,j)=quantile(net(i,j,:),0.25);
Q2(i,j)=quantile(net(i,j,:),0.5);
Q3(i,j)=quantile(net(i,j,:),0.75);
end
end

formatIn = 'dd/mm/yy';

% Events: 
tradewar=datenum('22/03/18',formatIn); 
lehman=datenum('15/09/08',formatIn);
greece2=datenum('21/07/11',formatIn); % 2nd bailout to Greece
brexit=datenum('23/06/16',formatIn); 

% Lehman Brothers:
for dd=1:1:size(Datestr,1)
    if datenum(Datestr(dd,:),formatIn)==lehman
       Z=dd; %to find out the point of the date
    end
end
pre=net(:,:,Z-1);   % outcome of the day before the event
on=net(:,:,Z);      % outcome of the day on the event
after1=net(:,:,Z+1); % outcome of the day after the event
after2=net(:,:,Z+2); % outcome of 2 days after the event

for i=1:1:K
for j=1:1:K
if pre(i,j)>=Q3(i,j)
   pre(i,j)=3;
elseif pre(i,j)>=Q2(i,j)
   pre(i,j)=1.5;
elseif pre(i,j)>=Q1(i,j) 
   pre(i,j)=0.3;
else
   pre(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if on(i,j)>=Q3(i,j)
   on(i,j)=3;
elseif on(i,j)>=Q2(i,j)
   on(i,j)=1.5;
elseif on(i,j)>=Q1(i,j) 
   on(i,j)=0.3;
else
   on(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if after1(i,j)>=Q3(i,j)
   after1(i,j)=3;
elseif after1(i,j)>=Q2(i,j)
   after1(i,j)=1.5;
elseif after1(i,j)>=Q1(i,j) 
   after1(i,j)=0.3;
else
   after1(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if after2(i,j)>=Q3(i,j)
   after2(i,j)=3;
elseif after2(i,j)>=Q2(i,j)
   after2(i,j)=1.5;
elseif after2(i,j)>=Q1(i,j) 
   after2(i,j)=0.3;
else
   after2(i,j)=0;

end
end
end

names = {'USA' 'UK' 'GER' 'FRA' 'IT' 'JPN' 'HK' 'CHN'};

figure(12)
subplot(2,2,1)
G1 = digraph(pre',names,'omitselfloops');
LWidths1 = G1.Edges.Weight;
plot(G1,'LineWidth',LWidths1,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(a) 12/09/2008';

subplot(2,2,2)
G2 = digraph(on',names,'omitselfloops');
LWidths2 = G2.Edges.Weight;
plot(G2,'LineWidth',LWidths2,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(b) 15/09/2008';

subplot(2,2,3)
G3 = digraph(after1',names,'omitselfloops');
LWidths3 = G3.Edges.Weight;
plot(G3,'LineWidth',LWidths3,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(c) 16/09/2008';

subplot(2,2,4)
G4 = digraph(after2',names,'omitselfloops');
LWidths4 = G4.Edges.Weight;
plot(G4,'LineWidth',LWidths4,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(d) 17/09/2008';

% Brexit: 23/06/16
for dd=1:1:size(Datestr,1)
    if datenum(Datestr(dd,:),formatIn)==brexit
       Z=dd; %to find out the point of the date
    end
end
bpre=net(:,:,Z-1);   % outcome of the day before the event
bon=net(:,:,Z);      % outcome of the day on the event
bafter1=net(:,:,Z+1); % outcome of the day after the event
bafter2=net(:,:,Z+2); % outcome of 2 days after the event

for i=1:1:K
for j=1:1:K
if bpre(i,j)>=Q3(i,j)
   bpre(i,j)=3;
elseif bpre(i,j)>=Q2(i,j)
   bpre(i,j)=1.5;
elseif bpre(i,j)>=Q1(i,j) 
   bpre(i,j)=0.3;
else
   bpre(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if bon(i,j)>=Q3(i,j)
   bon(i,j)=3;
elseif bon(i,j)>=Q2(i,j)
   bon(i,j)=1.5;
elseif bon(i,j)>=Q1(i,j) 
   bon(i,j)=0.3;
else
   bon(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if bafter1(i,j)>=Q3(i,j)
   bafter1(i,j)=3;
elseif bafter1(i,j)>=Q2(i,j)
   bafter1(i,j)=1.5;
elseif bafter1(i,j)>=Q1(i,j) 
   bafter1(i,j)=0.3;
else
   bafter1(i,j)=0;

end
end
end

for i=1:1:K
for j=1:1:K
if bafter2(i,j)>=Q3(i,j)
   bafter2(i,j)=3;
elseif bafter2(i,j)>=Q2(i,j)
   bafter2(i,j)=1.5;
elseif bafter2(i,j)>=Q1(i,j) 
   bafter2(i,j)=0.3;
else
   bafter2(i,j)=0;

end
end
end

names = {'USA' 'UK' 'GER' 'FRA' 'IT' 'JPN' 'HK' 'CHN'};

figure(13)
subplot(2,2,1)
G1 = digraph(bpre',names,'omitselfloops');
LWidths1 = G1.Edges.Weight;
plot(G1,'LineWidth',LWidths1,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(a) 22/06/2016';

subplot(2,2,2)
G2 = digraph(bon',names,'omitselfloops');
LWidths2 = G2.Edges.Weight;
plot(G2,'LineWidth',LWidths2,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(b) 23/06/2016';

subplot(2,2,3)
G3 = digraph(bafter1',names,'omitselfloops');
LWidths3 = G3.Edges.Weight;
plot(G3,'LineWidth',LWidths3,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(c) 24/06/2016';

subplot(2,2,4)
G4 = digraph(bafter2',names,'omitselfloops');
LWidths4 = G4.Edges.Weight;
plot(G4,'LineWidth',LWidths4,'EdgeColor','b','NodeColor','k','Layout','circle');
title '(d) 27/06/2016';


%% Total connectedness plot
figure(3)
plot(TC_out,'b');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
title 'Total connectedness';   % plot of toal connectedness


% To find out one specific day:
event=datenum('15/09/08',formatIn); % enter the date of an event here!!!
for dd=1:1:size(Datestr,1)
    if datenum(Datestr(dd,:),formatIn)==event
       Z=dd; %to find out the point of the date
    end
end
special=Datestr(2698,:); % enter data point in the first argument!!!




%% Total directional connectedness plot:

figure (5)
subplot(4,2,1)
plot(oc_out_usa, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_usa,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_usa,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'USA';
legend

subplot(4,2,2)
plot(oc_out_uk, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_uk,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_uk,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'UK';
legend

subplot(4,2,3)
plot(oc_out_deu, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_deu,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_deu,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'Germany';
legend

subplot(4,2,4)
plot(oc_out_fra, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_fra,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_fra,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'France';
legend

subplot(4,2,5)
plot(oc_out_itly, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_itly,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_itly,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'Italy';
legend

subplot(4,2,6)
plot(oc_out_jpn, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_jpn,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_jpn,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'Japan';
legend

subplot(4,2,7)
plot(oc_out_hk, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_hk,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_hk,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'Hong Kong';
legend

subplot(4,2,8)
plot(oc_out_chn, 'm', 'DisplayName', 'OC');
hold on;
plot(ic_out_chn,'c', 'DisplayName', 'IN');
hold on;
plot(ntdc_out_chn,'--k', 'DisplayName', 'NET');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
title 'China';
legend

figure (6)
plot(oc_out_uk,'r','DisplayName','UK');
hold on;
plot(oc_out_deu,'g','DisplayName','GER');
hold on;
plot(oc_out_fra,'b','DisplayName','FRA');
hold on;
plot(oc_out_itly,'c','DisplayName','IT');
hold on;
plot(oc_out_usa,'color',[0.4660 0.6740 0.1880],'DisplayName','USA');
hold on;
plot(oc_out_jpn,'k','DisplayName','JPN');
hold on;
plot(oc_out_hk,'m','DisplayName','HK');
hold on;
plot(oc_out_chn,'color',[0.9100    0.4100    0.1700],'DisplayName','CHN');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
% title 'Outgoing connectedness plot';
legend

figure (7)
plot(ic_out_uk,'r','DisplayName','UK');
hold on;
plot(ic_out_deu,'g','DisplayName','GER');
hold on;
plot(ic_out_fra,'b','DisplayName','FRA');
hold on;
plot(ic_out_itly,'c','DisplayName','IT');
hold on;
plot(ic_out_usa,'color',[0.4660 0.6740 0.1880],'DisplayName','USA');
hold on;
plot(ic_out_jpn,'k','DisplayName','JPN');
hold on;
plot(ic_out_hk,'m','DisplayName','HK');
hold on;
plot(ic_out_chn,'color',[0.9100    0.4100    0.1700],'DisplayName','CHN');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
% title 'Incoming connectedness plot';
legend

figure (8)
plot(ntdc_out_uk,'r','DisplayName','UK');
hold on;
plot(ntdc_out_deu,'g','DisplayName','GER');
hold on;
plot(ntdc_out_fra,'b','DisplayName','FRA');
hold on;
plot(ntdc_out_itly,'c','DisplayName','IT');
hold on;
plot(ntdc_out_usa,'color',[0.4660 0.6740 0.1880],'DisplayName','USA');
hold on;
plot(ntdc_out_jpn,'k','DisplayName','JPN');
hold on;
plot(ntdc_out_hk,'m','DisplayName','HK');
hold on;
plot(ntdc_out_chn,'color',[0.9100    0.4100    0.1700],'DisplayName','CHN');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
% title 'Net total connectedness plot';
legend


%% Pairwise connectedness:

% To plot pairwise connectedness and net pairwise connectedness, 
% first take out the time series of each element, name s12 for the (1,2) element of Stilde_out.
s12=zeros(1,size(Stilde_out,3)); s13=zeros(1,size(Stilde_out,3)); s14=zeros(1,size(Stilde_out,3));
s15=zeros(1,size(Stilde_out,3)); s16=zeros(1,size(Stilde_out,3)); s17=zeros(1,size(Stilde_out,3)); s18=zeros(1,size(Stilde_out,3));
s21=zeros(1,size(Stilde_out,3)); s23=zeros(1,size(Stilde_out,3)); s24=zeros(1,size(Stilde_out,3));
s25=zeros(1,size(Stilde_out,3)); s26=zeros(1,size(Stilde_out,3)); s27=zeros(1,size(Stilde_out,3)); s28=zeros(1,size(Stilde_out,3));
s31=zeros(1,size(Stilde_out,3)); s32=zeros(1,size(Stilde_out,3)); s34=zeros(1,size(Stilde_out,3));
s35=zeros(1,size(Stilde_out,3)); s36=zeros(1,size(Stilde_out,3)); s37=zeros(1,size(Stilde_out,3)); s38=zeros(1,size(Stilde_out,3));
s41=zeros(1,size(Stilde_out,3)); s42=zeros(1,size(Stilde_out,3)); s43=zeros(1,size(Stilde_out,3));
s45=zeros(1,size(Stilde_out,3)); s46=zeros(1,size(Stilde_out,3)); s47=zeros(1,size(Stilde_out,3)); s48=zeros(1,size(Stilde_out,3));
s51=zeros(1,size(Stilde_out,3)); s52=zeros(1,size(Stilde_out,3)); s53=zeros(1,size(Stilde_out,3));
s54=zeros(1,size(Stilde_out,3)); s56=zeros(1,size(Stilde_out,3)); s57=zeros(1,size(Stilde_out,3)); s58=zeros(1,size(Stilde_out,3));
s61=zeros(1,size(Stilde_out,3)); s62=zeros(1,size(Stilde_out,3)); s63=zeros(1,size(Stilde_out,3));
s64=zeros(1,size(Stilde_out,3)); s65=zeros(1,size(Stilde_out,3)); s67=zeros(1,size(Stilde_out,3)); s68=zeros(1,size(Stilde_out,3));
s71=zeros(1,size(Stilde_out,3)); s72=zeros(1,size(Stilde_out,3)); s73=zeros(1,size(Stilde_out,3));
s74=zeros(1,size(Stilde_out,3)); s75=zeros(1,size(Stilde_out,3)); s76=zeros(1,size(Stilde_out,3)); s78=zeros(1,size(Stilde_out,3));
s81=zeros(1,size(Stilde_out,3)); s82=zeros(1,size(Stilde_out,3)); s83=zeros(1,size(Stilde_out,3));
s84=zeros(1,size(Stilde_out,3)); s85=zeros(1,size(Stilde_out,3)); s86=zeros(1,size(Stilde_out,3)); s87=zeros(1,size(Stilde_out,3));


for t=1:1:size(Stilde_out,3)
s12(:,t)=Stilde_out(1,2,t); s13(:,t)=Stilde_out(1,3,t); s14(:,t)=Stilde_out(1,4,t);
s15(:,t)=Stilde_out(1,5,t); s16(:,t)=Stilde_out(1,6,t); s17(:,t)=Stilde_out(1,7,t); s18(:,t)=Stilde_out(1,8,t);
s21(:,t)=Stilde_out(2,1,t); s23(:,t)=Stilde_out(2,3,t); s24(:,t)=Stilde_out(2,4,t);
s25(:,t)=Stilde_out(2,5,t); s26(:,t)=Stilde_out(2,6,t); s27(:,t)=Stilde_out(2,7,t); s28(:,t)=Stilde_out(2,8,t);
s31(:,t)=Stilde_out(3,1,t); s32(:,t)=Stilde_out(3,2,t); s34(:,t)=Stilde_out(3,4,t);
s35(:,t)=Stilde_out(3,5,t); s36(:,t)=Stilde_out(3,6,t); s37(:,t)=Stilde_out(3,7,t); s38(:,t)=Stilde_out(3,8,t);
s41(:,t)=Stilde_out(4,1,t); s42(:,t)=Stilde_out(4,2,t); s43(:,t)=Stilde_out(4,3,t);
s45(:,t)=Stilde_out(4,5,t); s46(:,t)=Stilde_out(4,6,t); s47(:,t)=Stilde_out(4,7,t); s48(:,t)=Stilde_out(4,8,t);
s51(:,t)=Stilde_out(5,1,t); s52(:,t)=Stilde_out(5,2,t); s53(:,t)=Stilde_out(5,3,t);
s54(:,t)=Stilde_out(5,4,t); s56(:,t)=Stilde_out(5,6,t); s57(:,t)=Stilde_out(5,7,t); s58(:,t)=Stilde_out(5,8,t);
s61(:,t)=Stilde_out(6,1,t); s62(:,t)=Stilde_out(6,2,t); s63(:,t)=Stilde_out(6,3,t);
s64(:,t)=Stilde_out(6,4,t); s65(:,t)=Stilde_out(6,5,t); s67(:,t)=Stilde_out(6,7,t); s68(:,t)=Stilde_out(6,8,t);
s71(:,t)=Stilde_out(7,1,t); s72(:,t)=Stilde_out(7,2,t); s73(:,t)=Stilde_out(7,3,t);
s74(:,t)=Stilde_out(7,4,t); s75(:,t)=Stilde_out(7,5,t); s76(:,t)=Stilde_out(7,6,t); s78(:,t)=Stilde_out(7,8,t);
s81(:,t)=Stilde_out(8,1,t); s82(:,t)=Stilde_out(8,2,t); s83(:,t)=Stilde_out(8,3,t);
s84(:,t)=Stilde_out(8,4,t); s85(:,t)=Stilde_out(8,5,t); s86(:,t)=Stilde_out(8,6,t); s87(:,t)=Stilde_out(8,7,t);
end


c12=zeros(1,size(C_out,3)); c13=zeros(1,size(C_out,3)); c14=zeros(1,size(C_out,3));
c15=zeros(1,size(C_out,3)); c16=zeros(1,size(C_out,3)); c17=zeros(1,size(C_out,3)); c18=zeros(1,size(C_out,3));
c21=zeros(1,size(C_out,3)); c23=zeros(1,size(C_out,3)); c24=zeros(1,size(C_out,3));
c25=zeros(1,size(C_out,3)); c26=zeros(1,size(C_out,3)); c27=zeros(1,size(C_out,3)); c28=zeros(1,size(C_out,3));
c31=zeros(1,size(C_out,3)); c32=zeros(1,size(C_out,3)); c34=zeros(1,size(C_out,3));
c35=zeros(1,size(C_out,3)); c36=zeros(1,size(C_out,3)); c37=zeros(1,size(C_out,3)); c38=zeros(1,size(C_out,3));
c41=zeros(1,size(C_out,3)); c42=zeros(1,size(C_out,3)); c43=zeros(1,size(C_out,3));
c45=zeros(1,size(C_out,3)); c46=zeros(1,size(C_out,3)); c47=zeros(1,size(C_out,3)); c48=zeros(1,size(C_out,3));
c51=zeros(1,size(C_out,3)); c52=zeros(1,size(C_out,3)); c53=zeros(1,size(C_out,3));
c54=zeros(1,size(C_out,3)); c56=zeros(1,size(C_out,3)); c57=zeros(1,size(C_out,3)); c58=zeros(1,size(C_out,3));
c61=zeros(1,size(C_out,3)); c62=zeros(1,size(C_out,3)); c63=zeros(1,size(C_out,3));
c64=zeros(1,size(C_out,3)); c65=zeros(1,size(C_out,3)); c67=zeros(1,size(C_out,3)); c68=zeros(1,size(C_out,3));
c71=zeros(1,size(C_out,3)); c72=zeros(1,size(C_out,3)); c73=zeros(1,size(C_out,3));
c74=zeros(1,size(C_out,3)); c75=zeros(1,size(C_out,3)); c76=zeros(1,size(C_out,3)); c78=zeros(1,size(C_out,3));
c81=zeros(1,size(C_out,3)); c82=zeros(1,size(C_out,3)); c83=zeros(1,size(C_out,3));
c84=zeros(1,size(C_out,3)); c85=zeros(1,size(C_out,3)); c86=zeros(1,size(C_out,3)); c87=zeros(1,size(C_out,3));

for t=1:1:size(C_out,3)
c12(:,t)=C_out(1,2,t); c13(:,t)=C_out(1,3,t); c14(:,t)=C_out(1,4,t);
c15(:,t)=C_out(1,5,t); c16(:,t)=C_out(1,6,t); c17(:,t)=C_out(1,7,t); c18(:,t)=C_out(1,8,t);
c21(:,t)=C_out(2,1,t); c23(:,t)=C_out(2,3,t); c24(:,t)=C_out(2,4,t);
c25(:,t)=C_out(2,5,t); c26(:,t)=C_out(2,6,t); c27(:,t)=C_out(2,7,t); c28(:,t)=C_out(2,8,t);
c31(:,t)=C_out(3,1,t); c32(:,t)=C_out(3,2,t); c34(:,t)=C_out(3,4,t);
c35(:,t)=C_out(3,5,t); c36(:,t)=C_out(3,6,t); c37(:,t)=C_out(3,7,t); c38(:,t)=C_out(3,8,t);
c41(:,t)=C_out(4,1,t); c42(:,t)=C_out(4,2,t); c43(:,t)=C_out(4,3,t);
c45(:,t)=C_out(4,5,t); c46(:,t)=C_out(4,6,t); c47(:,t)=C_out(4,7,t); c48(:,t)=C_out(4,8,t);
c51(:,t)=C_out(5,1,t); c52(:,t)=C_out(5,2,t); c53(:,t)=C_out(5,3,t);
c54(:,t)=C_out(5,4,t); c56(:,t)=C_out(5,6,t); c57(:,t)=C_out(5,7,t); c58(:,t)=C_out(5,8,t);
c61(:,t)=C_out(6,1,t); c62(:,t)=C_out(6,2,t); c63(:,t)=C_out(6,3,t);
c64(:,t)=C_out(6,4,t); c65(:,t)=C_out(6,5,t); c67(:,t)=C_out(6,7,t); c68(:,t)=C_out(6,8,t);
c71(:,t)=C_out(7,1,t); c72(:,t)=C_out(7,2,t); c73(:,t)=C_out(7,3,t);
c74(:,t)=C_out(7,4,t); c75(:,t)=C_out(7,5,t); c76(:,t)=C_out(7,6,t); c78(:,t)=C_out(7,8,t);
c81(:,t)=C_out(8,1,t); c82(:,t)=C_out(8,2,t); c83(:,t)=C_out(8,3,t);
c84(:,t)=C_out(8,4,t); c85(:,t)=C_out(8,5,t); c86(:,t)=C_out(8,6,t); c87(:,t)=C_out(8,7,t);     
end

d=574;
figure(20)
subplot(3,2,1)
plot(s23,'k','DisplayName','GER to UK');
hold on;
plot(s32,'r','DisplayName','UK to GER');
hold on;
plot(c23,'r--','DisplayName','Net UK to GER');
hold on;
plot(c32,'k--','DisplayName','Net GER to UK');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'Germany vs UK'

subplot(3,2,2)
plot(s24,'b','DisplayName','FRA to UK');
hold on;
plot(s42,'r','DisplayName','UK to FRA');
hold on;
plot(c24,'r--','DisplayName','Net UK to FRA');
hold on;
plot(c42,'b--','DisplayName','Net FRA to UK');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'France vs UK'

subplot(3,2,3)
plot(s25,'g','DisplayName','IT to UK');
hold on;
plot(s52,'r','DisplayName','UK to IT');
hold on;
plot(c25,'r--','DisplayName','Net UK to IT');
hold on;
plot(c52,'g--','DisplayName','Net IT to UK');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'Italy vs UK'

subplot(3,2,4)
plot(s34,'b','DisplayName','FRA to GER');
hold on;
plot(s43,'k','DisplayName','GER to FRA');
hold on;
plot(c34,'k--','DisplayName','NEt GER to FRA');
hold on;
plot(c43,'b--','DisplayName','Net FRA to GER');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'France vs Germany'

subplot(3,2,5)
plot(s35,'g','DisplayName','IT to GER');
hold on;
plot(s53,'k','DisplayName','GER to IT');
hold on;
plot(c35,'k--','DisplayName','Net GER to IT');
hold on;
plot(c53,'g--','DisplayName','Net IT to GER');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'Italy vs Germany'

subplot(3,2,6)
plot(s45,'g','DisplayName','IT to FRA');
hold on;
plot(s54,'b','DisplayName','FRA to IT');
hold on;
plot(c45,'b--','DisplayName','Net FRA to IT');
hold on;
plot(c54,'g--','DisplayName','Net IT to FRA');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'France vs Italy'


figure(22)
subplot(3,2,1)
plot(s16,'color','[0.9290 0.6940 0.1250]','DisplayName','JPN to USA');
hold on;
plot(s61,'color','[0 0.4470 0.7410]','DisplayName','USA to JPN');
hold on;
plot(c16,'color','[0 0.4470 0.7410]','LineStyle','--','DisplayName','Net USA to JPN');
hold on;
plot(c61,'color','[0.9290 0.6940 0.1250]','LineStyle','--','DisplayName','Net JPN to USA');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'USA vs Japan'

subplot(3,2,2)
plot(s17,'color','[0.4940 0.1840 0.5560]','DisplayName','HK to USA');
hold on;
plot(s71,'color','[0 0.4470 0.7410]','DisplayName','USA to HK');
hold on;
plot(c17,'color','[0 0.4470 0.7410]','LineStyle','--','DisplayName','Net USA to HK');
hold on;
plot(c71,'color','[0.4940 0.1840 0.5560]','LineStyle','--','DisplayName','Net HK to USA');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'USA vs Hong Kong'

subplot(3,2,3)
plot(s18,'color','[0.6350 0.0780 0.1840]','DisplayName','CHN to USA');
hold on;
plot(s81,'color','[0 0.4470 0.7410]','DisplayName','USA to CHN');
hold on;
plot(c18,'color','[0 0.4470 0.7410]','LineStyle','--','DisplayName','Net USA to CHN');
hold on;
plot(c81,'color','[0.6350 0.0780 0.1840]','LineStyle','--','DisplayName','Net CHN to USA');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'USA vs China'

subplot(3,2,4)
plot(s67,'color','[0.4940 0.1840 0.5560]','DisplayName','HK to JPN');
hold on;
plot(s76,'color','[0.9290 0.6940 0.1250]','DisplayName','JPN to HK');
hold on;
plot(c67,'color','[0.9290 0.6940 0.1250]','LineStyle','--','DisplayName','NEt JPN to HK');
hold on;
plot(c76,'color','[0.4940 0.1840 0.5560]','LineStyle','--','DisplayName','Net HK to JPN');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'Hong Kong vs Japan'

subplot(3,2,5)
plot(s68,'color','[0.6350 0.0780 0.1840]','DisplayName','CHN to JPN');
hold on;
plot(s86,'color','[0.9290 0.6940 0.1250]','DisplayName','JPN to CHN');
hold on;
plot(c68,'color','[0.9290 0.6940 0.1250]','LineStyle','--','DisplayName','Net JPN to CHN');
hold on;
plot(c86,'color','[0.6350 0.0780 0.1840]','LineStyle','--','DisplayName','Net CHN to JPN');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'China vs Japan'

subplot(3,2,6)
plot(s78,'color','[0.6350 0.0780 0.1840]','DisplayName','CHN to HK');
hold on;
plot(s87,'color','[0.4940 0.1840 0.5560]','DisplayName','HK to CHN');
hold on;
plot(c78,'color','[0.4940 0.1840 0.5560]','LineStyle','--','DisplayName','Net HK to CHN');
hold on;
plot(c87,'color','[0.6350 0.0780 0.1840]','LineStyle','--','DisplayName','Net CHN to HK');
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,4:end);
hold off;
legend
title 'China vs Hong Kong'


% d =205;
% figure(21) % CHN vs USA
% plot(s18,'color',[0 0.4470 0.7410],'DisplayName','CHN to USA');
% hold on;
% plot(s81,'color',[0.4940 0.1840 0.5560],'DisplayName','USA to CHN');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend
% 
% figure(22)  % CHN vs HK
% plot(s78,'color',[0 0.4470 0.7410],'DisplayName','CHN to HK');
% hold on;
% plot(s87,'color',[0.6350 0.0780 0.1840],'DisplayName','HK to CHN');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend

% figure(12)
% subplot(4,1,1)
% plot(s23,'r','DisplayName','GER to UK');
% hold on;
% plot(s24,'g','DisplayName','FRA to UK');
% hold on;
% plot(s25,'b','DisplayName','IT to UK');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend
% title 'To UK'
% 
% subplot(4,1,2)
% plot(s32,'r','DisplayName','UK to GER');
% hold on;
% plot(s34,'g','DisplayName','FRA to GER');
% hold on;
% plot(s35,'b','DisplayName','IT to GER');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend
% title 'To Germany'
% 
% subplot(4,1,3)
% plot(s42,'r','DisplayName','UK to FRA');
% hold on;
% plot(s43,'g','DisplayName','GER to FRA');
% hold on;
% plot(s45,'b','DisplayName','IT to FRA');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend
% title 'To France'
% 
% subplot(4,1,4)
% plot(s52,'r','DisplayName','UK to IT');
% hold on;
% plot(s53,'g','DisplayName','GER to IT');
% hold on;
% plot(s54,'b','DisplayName','FRA to IT');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% hold off;
% legend
% title 'To Italy'

d=574;
figure(11)
subplot(K,K-1,1)
plot(s12);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to USA'

subplot(K,K-1,2)
plot(s13);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to USA'

subplot(K,K-1,3)
plot(s14);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to USA'

subplot(K,K-1,4)
plot(s15);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to USA'

subplot(K,K-1,5)
plot(s16);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to USA'

subplot(K,K-1,6)
plot(s17);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to USA'

subplot(K,K-1,7)
plot(s18);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to USA'

subplot(K,K-1,8)
plot(s21);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to UK'

subplot(K,K-1,9)
plot(s23);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to UK'

subplot(K,K-1,10)
plot(s24);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to UK'

subplot(K,K-1,11)
plot(s25);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to UK'

subplot(K,K-1,12)
plot(s26);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to UK'

subplot(K,K-1,13)
plot(s27);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to UK'

subplot(K,K-1,14)
plot(s28);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to UK'

subplot(K,K-1,15)
plot(s31);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to GER'

subplot(K,K-1,16)
plot(s32);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to GER'

subplot(K,K-1,17)
plot(s34);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to GER'

subplot(K,K-1,18)
plot(s35);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to GER'

subplot(K,K-1,19)
plot(s36);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to GER'

subplot(K,K-1,20)
plot(s37);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to GER'

subplot(K,K-1,21)
plot(s38);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to GER'

subplot(K,K-1,22)
plot(s41);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to FRA'

subplot(K,K-1,23)
plot(s42);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to FRA'

subplot(K,K-1,24)
plot(s43);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to FRA'

subplot(K,K-1,25)
plot(s45);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to FRA'

subplot(K,K-1,26)
plot(s46);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to FRA'

subplot(K,K-1,27)
plot(s47);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to FRA'

subplot(K,K-1,28)
plot(s48);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to FRA'

subplot(K,K-1,29)
plot(s51);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to IT'

subplot(K,K-1,30)
plot(s52);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to IT'

subplot(K,K-1,31)
plot(s53);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to IT'

subplot(K,K-1,32)
plot(s54);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to IT'

subplot(K,K-1,33)
plot(s56);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to IT'

subplot(K,K-1,34)
plot(s57);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to IT'

subplot(K,K-1,35)
plot(s58);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to IT'

subplot(K,K-1,36)
plot(s61);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to JPN'

subplot(K,K-1,37)
plot(s62);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to JPN'

subplot(K,K-1,38)
plot(s63);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to JPN'

subplot(K,K-1,39)
plot(s64);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to JPN'

subplot(K,K-1,40)
plot(s65);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to JPN'

subplot(K,K-1,41)
plot(s67);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to JPN'

subplot(K,K-1,42)
plot(s68);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to JPN'

subplot(K,K-1,43)
plot(s71);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to HK'

subplot(K,K-1,44)
plot(s72);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to HK'

subplot(K,K-1,45)
plot(s73);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to HK'

subplot(K,K-1,46)
plot(s74);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to HK'

subplot(K,K-1,47)
plot(s75);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to HK'

subplot(K,K-1,48)
plot(s76);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to HK'

subplot(K,K-1,49)
plot(s78);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'CHN to HK'

subplot(K,K-1,50)
plot(s81);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'USA to CHN'

subplot(K,K-1,51)
plot(s82);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'UK to CHN'

subplot(K,K-1,52)
plot(s83);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'GER to CHN'

subplot(K,K-1,53)
plot(s84);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'FRA to CHN'

subplot(K,K-1,54)
plot(s85);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'IT to CHN'

subplot(K,K-1,55)
plot(s86);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'JPN to CHN'

subplot(K,K-1,56)
plot(s87);
h1 = gca;
h1.XLim = [0,D];
h1.XTick = 1:d:D;
h1.XTickLabel = Datestr(1:d:D,7:end);
title 'HK to CHN'









 
% figure(9)
% subplot(2,2,1)
% plot(diff_itly,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'Italy';
% 
% 
% subplot(2,2,2)
% plot(diff_uk,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'UK';
% 
% 
% subplot(2,2,3)
% plot(diff_deu,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'Germany';
% 
% 
% subplot(2,2,4)
% plot(diff_fra,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'France';
% %title 'Relative difference of outgoing connectedness'; % plot of relative difference of outgoing connectedness of Italy, UK, Germany and France
% 
% figure(10)
% subplot(2,2,1)
% plot(diff_usa,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'USA';
% 
% 
% subplot(2,2,2)
% plot(diff_jpn,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'Japan';
% 
% 
% subplot(2,2,3)
% plot(diff_hk,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'Hong Kong';
% 
% 
% subplot(2,2,4)
% plot(diff_chn,'g');
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% title 'China';
% %title 'Relative difference of outgoing connectedness'; % plot of Relative difference of outgoing connectedness of USA,JPN,HK,CHN
% 
% figure (11)
% plot(AD);
% h1 = gca;
% h1.XLim = [0,D];
% h1.XTick = 1:d:D;
% h1.XTickLabel = Datestr(1:d:D,4:end);
% % title 'Aggregated relative difference of country-wise out-of-sample and in-sample relative connectedness';



