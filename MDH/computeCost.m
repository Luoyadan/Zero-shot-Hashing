function [ cost ] = computeCost(Y,R,W,B,P,X,L,lambda,nu,epsilon,delta)
%COMPUTECOST �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
cost = norm(R'*Y-W'*B,'fro')^2+lambda*norm(W,'fro')^2+nu*norm(B-P'*X,'fro')^2+epsilon*norm(P,'fro')^2+delta*trace(R'*Y*L*Y'*R);


end

