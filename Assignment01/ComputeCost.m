function J = ComputeCost(X, Y, W, b, lambda)
% Input  -  X       (dxn)
%        -  Y       (Kxn)
%        -  W       (Kxd)
%        -  b       (Kx1)
%        -  lambda  (1x1)
% Output -  J       (1x1)

P = EvaluateClassifier(X, W, b);
loss = sum(diag(-log(Y'*P)))/size(X, 2);
J = loss + lambda*sum(sum(W.^2));
end
