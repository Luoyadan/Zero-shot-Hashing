function [G,F,Z,R] = mySVM_binary(X,y,Z,gmap,Fmap,Rmap,tol,maxItr,debug)
    options.nu = Fmap.nu;
    options.lambda = gmap.lambda;
    options.debug = debug;
    options.tol = tol;
    options.l = 64;
    
    options.epsilon = 1e-3;
    options.delta = 1e-2;
    options.k = 10;
    options.maxItr = 10;
    options.orth = 1;
    options.rotate = 1;
    [W, P, B, R] = mdh(X',y',options);
    F.W = P;
    G.W = W;
    Z = B;
end