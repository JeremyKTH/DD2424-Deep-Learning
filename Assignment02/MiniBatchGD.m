function [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch, W, b, eta, lambda)
% Input  -  Xbatch    (dxn) - for training
%        -  Ybatch    (1xn)
%        -  W         (Kxd)
%        -  b         (Kx1)
%        -  eta       (1x1)
%        -  lambda    (1x1)
% Output -  W*        (Kxd)
%        -  b*        (Kx1)

% Compute gradients
[grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
        
% Update weight & bias
W{1} = W{1} - eta*grad_W{1};
b{1} = b{1} - eta*grad_b{1};
W{2} = W{2} - eta*grad_W{2};
b{2} = b{2} - eta*grad_b{2};

Wstar = W;
bstar = b;
end
