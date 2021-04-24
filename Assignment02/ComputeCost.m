function [cost, loss] = ComputeCost(X, Y, W, b, lambda)
% Input  -  X       (dxn)
%        -  Y       (Kxn)
%        -  W1      (mxd)
%        -  W2      (Kxm)
%        -  b1      (mx1)
%        -  b2      (Kx1)
%        -  lambda  (1x1)
% Output -  cost    (1x1)
%        -  loss    (1x1)

n = size(Y,2);
W1 = W{1};
W2 = W{2};

[ForwardPass] = EvaluateClassifier(X, W, b);
P = ForwardPass{2};

l_cross = -log(Y'*P);
loss = (1/n)*(trace(l_cross));
cost = loss + lambda*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
end
