function [cost, loss] = ComputeCost(X, Y, NetParams, lambda, train)
% Input  -  X         (dxn)
%        -  Y         (Kxn)
%        -  NetParams (struct) - {W, b, use_bn}
%        -  lambda    (1x1)
%        -  train     (1x1)
% Output -  cost      (1x1)
%        -  loss      (1x1)

n = size(Y,2);
W = NetParams.W;

% Cross-entropy
ForwardPass = EvaluateClassifier(X, NetParams, train);
P = ForwardPass.P;

% L2 regression
penalty = 0;
for i = 1: numel(W)
    penalty = penalty + sum(sum(W{i}.^2));
end

l_cross = -log(Y'*P);
loss = (1/n)*(trace(l_cross));
cost = loss + lambda*penalty;
end
