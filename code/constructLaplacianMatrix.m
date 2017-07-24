function [ L,Ln ] = constructLaplacianMatrix( X, k )
% selftune the variance in Gaussian kernel function
% addpath ./codes/self-tuning/;

if ~exist('k','var')
    k = 5;
end

AA = EuDist2(X');
n = size(AA, 1);
[dumb, idx] = sort(AA, 2); % sort each row
A = zeros(n);
for i = 1:n
    A(i, idx(i,2:k+1)) = AA(i, idx(i,2:k+1));
end;
A = max(A, A');
clear AA idx dumb;

% Find the count of nonzero for each column
col_count = sum(A~=0, 1)';
col_sum = sum(A, 1)';
col_mean = col_sum ./ col_count;
[x y val] = find(A);
A = sparse(x, y, -val.*val./col_mean(x)./col_mean(y)./2);
clear col_count col_sum col_mean x y val;
% Do exp function sequentially because of memory limitation
num = 2000;
num_iter = ceil(n/num);
S = sparse([]);
for i = 1:num_iter
    start_index = 1 + (i-1)*num;
    end_index = min(i*num, n);
    S1 = spfun(@exp, A(:,start_index:end_index)); % sparse exponential func
    S = [S S1];
    clear S1;
end
A = S;
clear S;

A = sparse(A);

% construct the Laplacian matrix using the Gaussian kernel function
D = diag(sum(A,2));
L = D - A;
% construct the normalized Laplacian matrix using the Gaussian kernel function
Dd = diag(D)+1e-12;
Dn=diag(sqrt(1./Dd)); Dn = sparse(Dn);
An = Dn*A*Dn; An = (An+An')/2;
Ln=speye(size(A,1)) - An;
Ln = (Ln+Ln')/2;

end

