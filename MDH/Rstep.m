function [R] = Rstep(W,L,Y,Z,delta)
R = rand(size(Y,2));

R = (Y'*Y+delta*Y'*L*Y)\(Y'*Z*W);

end

