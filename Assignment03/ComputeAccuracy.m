function acc = ComputeAccuracy(X, y, NetParams, train)
% Input  -  X           (dxn)
%        -  y           (1xn)
%        -  train       (1x1)
%        -  NetParams   (struct)
% Output -  acc         (1x1)

% if NetParams.use_bn
%     mu = NetParams.mu_avg;
%     v  = NetParams.v_avg;
%     ForwardPass = EvaluateClassifier(X, NetParams, mu, v);
%     P = ForwardPass.P;
% else
%     ForwardPass = EvaluateClassifier(X, NetParams);
%     P = ForwardPass.P;
% end


ForwardPass = EvaluateClassifier(X, NetParams, train);
P = ForwardPass.P;

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