opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

X = rand(20,100);
Y = randi([0 1],100,5);Y = 2*Y-1;
W = rand(10,5);
U0 = randn(20,10);    U0 = orth(U0);

tic; [X, out]= OptStiefelGBB(U0, @test_fun, opts, X,Y,W); tsolve = toc;