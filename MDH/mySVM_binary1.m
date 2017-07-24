function [G, F, Z] = mySVM_binary(X,y,Z,gmap,Fmap,Rmap,tol,maxItr,debug)

% ---------- Argument defaults ----------
if ~exist('debug','var') || isempty(debug)
    debug=1;
end
if ~exist('tol','var') || isempty(tol)
    tol=1e-5;
end
if ~exist('maxItr','var') || isempty(maxItr)
    maxItr=1000;
end
nu = Fmap.nu;
delta = 1e5;
% ---------- End ----------

% label matrix N x c
if isvector(y) 
    Y = sparse(1:length(y), double(y), 1); Y = full(Y);    
else
    Y = y;
    y = find(Y);
end

% initial W in G-step and similarity matrix
G.W = rand(size(Z,2),size(Y,2));

addpath code\;
S = constructW(X);
L = diag(sum(S))-S;
   
%R-step
R = Rstep(G.W,L,Y,Z,Rmap.delta);
% g-step
switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(Z, Y*R, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'
        svm_option = ['-q -s 4 -c ', num2str(1/gmap.lambda)];
        model = train(double(y),sparse(Z),svm_option);
        Wg = model.w';
end
G.W = Wg;

% F-step
if Fmap.sigmoid
    maxItr_F = 100; % for LBFGS optimization
    [WF, ~] = FOpt(X, Z, Fmap.lambda, maxItr_F);
else
    [WF, ~, ~] = RRC(X, Z, Fmap.lambda);
end
F.W = WF; F.nu = nu;


i = 0; %E = Eg+EF;
while i < maxItr    
    i=i+1;  
    
    if debug,fprintf('Iteration  %03d: ',i);end
    
    % Z-step
    if  Fmap.sigmoid
        XF = sig(X*WF);
    else
        XF = X*WF;
    end
    switch gmap.loss
        case 'L2'
            Q = nu*XF + (Y*R)*Wg';
            
            Z = zeros(size(Z));          
            for time = 1:10           
               Z0 = Z;
                for k = 1 : size(Z,2)
                    Zk = Z; Zk(:,k) = [];
                    Wkk = Wg(k,:); Wk = Wg; Wk(k,:) = [];                    
                    Z(:,k) = sign(Q(:,k) -  Zk*Wk*Wkk');
                end
                
                if norm(Z-Z0,'fro') < 1e-6 * norm(Z0,'fro')
                    break
                end
            end
        case 'Hinge' 
            
            for ix_z = 1 : size(Z,1)
                
                w_ix_z = bsxfun(@minus, Wg(:,y(ix_Z)), Wg);
                Z(ix_z,:) = sign(2*nu*XF(ix_z,:) + delta*sum(w_ix_z,2)');
            end
             
    end
    
%     EZ = norm(Y-Z*Wg,'fro')^2 + nu*norm(Z-XF,'fro')^2
    
    
    R = Rstep(G.W,L,Y,Z,Rmap.delta);
    % g-step
    switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(Z, Y*R, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'        
        model = train(double(y),sparse(Z),svm_option);
        Wg = model.w';
    end
    G.W = Wg;
    
    % F-step 
    WF0 = WF;
    if Fmap.sigmoid
        maxItr_F = 100;
        [WF, ~] = FOpt(X, Z, Fmap.lambda, maxItr_F);
    else
        [WF, ~, ~] = RRC(X, Z, Fmap.lambda);
    end
    F.W = WF; F.nu = nu;
    
    
    
    if Fmap.sigmoid
        bias = norm(Z-sig(X*WF),'fro');
    else
        bias = norm(Z-X*WF,'fro');
    end
    
    obj_val(i)=computeCost(Y,R,G.W,Z,F.W,X,L,gmap.lambda,Fmap.nu,Fmap.lambda,Rmap.delta);
    if debug
        fprintf('  bias=%g\n',bias); 
        fprintf('  cost=%g\n',obj_val(i));
    end
    
%     if bias < tol*norm(Z,'fro')
%             break;
%     end 
%     
%     
%     if norm(WF-WF0,'fro') < tol * norm(WF0)
%         break;
%     end
    
%     if (abs(E(end-1)-E(end)) < tol*abs(E(end-1)))
%         beta=1.5*beta;        
%     end
    
end
if debug
    plot(obj_val);
end