function P = EvaluateClassifier(X, W, b)
% Input  -  X  (dxn)
%        -  W  (Kxd)
%        -  b  (Kx1)
% Output -  P  (Kxn)

n = size(X,2); % return the length of second dimension (100)
b = repmat(b,1,n); % create a matrix with copies of b (dim: 1x100)
s = W*X+b;
P = softmax(s);
end