function [ cost ] = computeCost(Y,R,W,B,P,X,L,lambda,nu,epsilon,delta)
%COMPUTECOST 此处显示有关此函数的摘要
%   此处显示详细说明
cost = norm(R'*Y-W'*B,'fro')^2+lambda*norm(W,'fro')^2+nu*norm(B-P'*X,'fro')^2+epsilon*norm(P,'fro')^2+delta*trace(R'*Y*L*Y'*R);


end

