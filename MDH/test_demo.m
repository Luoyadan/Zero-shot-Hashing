clear;clc; warning off all;

n = 1000;
d = 25;
c = 5;
l = 16;

X = rand(d,n);
Y = randi([0,1],c,n);

options.maxItr = 100;
options.debug = 1;
options.tol = 1e-3;
options.nu = 0.1;
options.lambda = 0.1;
options.delta = 0.1;
options.epsilon = 0.1;
options.l = l;

[W, P, B, R,obj_val] = mdh(X,Y,options);

