function [ForwardPass] = EvaluateClassifier(X, W, b)
% Input  -  X    (dxn)
%        -  W1   (mxd)
%        -  W2   (Kxm)
%        -  b1   (mx1)
%        -  b2   (Kx1)
% Output -  P    (Kxn)
%        -  h    (Kxn)

W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};
n_b = size(X,2);  % n_batch

% Input -> hidden layer
s1 = W1*X + b1*ones(1,n_b);  % add b1 to each column.

% reLu
h = max(s1,0);

% hidden -> output layer
P = softmax(W2*h + b2*ones(1,n_b));

ForwardPass = {};
ForwardPass{1} = h;   % reLu
ForwardPass{2} = P;   % probability
end
