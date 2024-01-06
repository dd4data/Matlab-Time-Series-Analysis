function [vbar, Abar, J]=tr2VAR1(y,B,p,inc)
% input: y is the whole time series data observed;
%        B is the VAR(p) coefficient matrix 
%        (KxKp without intercept; Kx(Kp+1) with intercept);
%        p is the specified lag order;
%        inc=1 indicates the VAR(p) includes an intercept;
% output: vbar is the intercept in the VAR(1) (Kpx1)
%         Abar is the coefficient matrix in the VAR(1) (KpxKp)
%         J is a KxKp matrix such that VAR(p) can be obtained from the VAR(1)

K=size(y,1);  % number of total observations

if inc==1
    vbar=[B(:,1);zeros(K*(p-1),1)];
    Abar=[B(:,2:end);eye(K*(p-1)),zeros(K*(p-1),K)];
else
    vbar=[];
    Abar=[B(:,1:end);eye(K*(p-1)),zeros(K*(p-1),K)];
end
J=[eye(K), zeros(K,K*(p-1))];

