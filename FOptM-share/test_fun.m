function [F,G] = test_fun(U,X,Y,W)
G = 2*((X*X')*U*(W*W')-X*Y*W');
F = sum(sum((Y - X'*U*W).^2));
end