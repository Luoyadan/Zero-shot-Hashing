function [F,G] = solve_R(R,W,B,Y,YY,YLY,delta)
WB = W'*B;
G = 2*((YY+delta*YLY)*R-Y*WB');
F = sum(sum((R'*Y - WB).^2))+delta*trace(R'*YLY*R);
end