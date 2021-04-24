function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
% Input  -  X       (dxn)
%        -  Y       (Kxn)
%        -  W       (Kxd)
%        -  lambda  (1x1)
% Output -  grad_w  (Kxd)
%        -  grad_b  (Kx1)
% 
g = -(Y-P)';
grad_W = (g'*X')/(size(X, 2)) + 2*lambda*W;
grad_b = (sum(g, 1)')/(size(X, 2));
end
