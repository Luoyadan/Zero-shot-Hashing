function [W, P, B, R,obj_val] = mdh(X,Y,options)

% ---------- Argument defaults ----------
if ~isfield(options,'debug') || isempty(options.debug)
    debug=1;
else
    debug = options.debug;
end
if ~isfield(options,'tol') || isempty(options.tol)
    tol=1e-5;
else
    tol = options.tol;
end
if ~isfield(options,'maxItr') || isempty(options.maxItr)
    maxItr=100;
else
    maxItr = options.maxItr;
end
if ~isfield(options,'lambda') || isempty(options.lambda)
    lambda=1;
else
    lambda = options.lambda;
end
if ~isfield(options,'delta') || isempty(options.delta)
    delta=1;
else
    delta = options.delta;
end
if ~isfield(options,'nu') || isempty(options.nu)
    nu = 1;
else
    nu = options.nu;
end
if ~isfield(options,'epsilon') || isempty(options.epsilon)
    epsilon = 1;
else
    epsilon = options.epsilon;
end

if ~isfield(options,'l') || isempty(options.l)
    l = 16;
else
    l = options.l;
end


if ~isfield(options,'k') || isempty(options.k)
    k = 5;
else
    k = options.k;
end


if ~isfield(options,'rotate') || isempty(options.rotate)
    rotate = 1;
else
    rotate = options.rotate;
end

if ~isfield(options,'orth') || isempty(options.orth)
    isorth = 1;
else
    isorth = options.orth;
end

% label matrix N x c
if isvector(Y) 
    Y = sparse(1:length(Y), double(Y), 1); Y = full(Y);    Y=Y';
end
[c,n] = size(Y);
d = size(X,1);

% initial W,P,B,R
W = rand(l,c);
randn('seed',3);
B=sign(randn(l,n));
if rotate
    R = rand(c,c);
    if isorth
        addpath FOptM-share\;
        R = orth(R);
    end
else
    R = eye(c);
    delta = 0;
end
% compute Laplacian matrix
addpath code\;
opts.k = k;
L = computeLaplacian(X',opts);
i = 0;
YY = Y*Y';
YLY = Y*L*Y';
XX = X*X';
while i < maxItr    
    i=i+1;  
    if debug,fprintf('Iteration  %03d: ',i);end

    % 1. Update P
    P=(XX+epsilon/nu*eye(d))\(X*B');
    
    % 2. Update B
    RY = R'*Y;
    Q = nu*P'*X + W*RY;
    Z = B;
    for time = 1
        for k = 1 : size(Z,1)
            Zk = Z; Zk(k,:) = [];
            Wkk = W(k,:); Wk = W; Wk(k,:) = [];
            Z(k,:) = sign(Q(k,:) -  (Zk'*Wk*Wkk')');
        end
    end
    B = Z;
    
    % 3. Update R
    if rotate
        if isorth
            opts.record = 0; %
            opts.mxitr  = 10;
            opts.xtol = 1e-3;
            opts.gtol = 1e-3;
            opts.ftol = 1e-3;
            R = OptStiefelGBB(R, @solve_R, [], W,B,Y,YY,YLY,delta);
        else
            R = (YY+delta*YLY)\(Y*B'*W);
        end
    end
        
    % 4. Update W
    W = (B*B'+lambda*eye(l))\(B*Y'*R);
    
    if debug
        obj_val(i)=computeCost(Y,R,W,B,P,X,L,lambda,nu,epsilon,delta);
        fprintf('cost=%g\n',obj_val(i));
    end
end
if debug
    plot(obj_val);
end