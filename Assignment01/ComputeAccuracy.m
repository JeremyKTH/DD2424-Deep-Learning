function acc = ComputeAccuracy(X, y, W, b)
% Input  -  X    (dxn)
%        -  y    (1xn)
%        -  W    (Kxd)
%        -  b    (Kx1)
% Output -  acc  (1x1)

P = EvaluateClassifier(X, W, b);
n = size(y, 2);
correct = 0;

for i = 1 : n
    [~, index] = max(P(:,i)); 
    if index == y(i)
        correct = correct + 1;
    end
end
acc = correct / n;
end
