function [ Ln,L ] = computeLaplacian( X,options )
    if (~exist('options','var'))
        options = [];
    end
    W = constructW(X,options);
    D = diag(sum(W,2));
    L = D - W;
    L = (L + L')/2;
    A = W;
    
    Dd = diag(D)+1e-12;
    Dn=diag(sqrt(1./Dd)); Dn = sparse(Dn);
    An = Dn*W*Dn; An = (An+An')/2;
    Ln=speye(size(W,1)) - An;


end

