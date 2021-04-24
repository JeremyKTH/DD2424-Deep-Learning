function [W, b] = init_param()
% Input  -  trainX (dxn)
% Output -  W1     (mxd)
%        -  W2     (Kxm)
%        -  b1     (mx1)
%        -  b2     (Kx1)
K = 10;            % image classes
d = 3072;          % image dimensions
m = 50;            % nodes in hidden layer

W1 = (1/sqrt(d)).*randn(m, d);  % weight1 - Gaussian distribution, mean = 1/sqrt(d), std = 0.01
W2 = (1/sqrt(m)).*randn(K, m);  % weight2 - Gaussian distribution, mean = 1/sqrt(m), std = 0.01
b1 = zeros(m, 1);
b2 = zeros(K, 1);

W = {W1, W2};
b = {b1, b2};
end
