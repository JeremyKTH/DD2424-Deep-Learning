function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
% Input  -  X           (dxn)
%        -  Y           (Kxn)
%        -  W1          (mxd)
%        -  W2          (Kxm)
%        -  b1          (mx1)
%        -  b2          (Kx1)
%        -  lambda      (1x1)
% Output -  grad_W1     (mxd)
%        -  grad_W2     (Kxm)
%        -  grad_b1     (mx1)
%        -  grad_b2     (Kx1)

W1 = W{1}; W2 = W{2};
n_b = size(X,2);     % batch size

% Forward Pass
[ForwardPass] = EvaluateClassifier(X,W,b);
P = ForwardPass{2};  % Probability
H = ForwardPass{1};  % reLu activation value

% Output -> hidden layer
G_b = -(Y-P);

grad_W2 = (1/n_b)*(G_b*H') + (2*lambda*W2);
grad_b2 = (1/n_b)*G_b*ones(n_b,1);

% hidden -> Input layer
G_b = W2'*G_b;
G_b = G_b.*sign(H); % reLu

grad_W1 = (1/n_b)*G_b*X'+ (2*lambda*W1);
grad_b1 = (1/n_b)*G_b*ones(n_b,1);


grad_W = {grad_W1, grad_W2};
grad_b = {grad_b1, grad_b2};
end
