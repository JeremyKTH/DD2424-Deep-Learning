function [gradcheck_W, gradcheck_b] = CheckGradAcc(ngrads, Grads)
% Input  -  ngrads      (mxn)   nodes x batch size
%        -  Grads       (mxn)   nodes x batch size
% Output -  gradcheck_W (1x1)    - max error
%        -  gradcheck_b (1x1)    - max error

% Numerical
ngrad_W = ngrads.W;
ngrad_b = ngrads.b;


% Analytical
grad_W = Grads.W;
grad_b = Grads.b;


% Maximum error
gradcheck_W = 0;
gradcheck_b = 0;


for i = 1 : numel(ngrad_W)
    temp_W = max(max(abs(ngrad_W{i} - grad_W{i})./max(0.01, abs(ngrad_W{i}) + abs(grad_W{i}))));
    temp_b = max(max(abs(ngrad_b{i} - grad_b{i})./max(0.01, abs(ngrad_b{i}) + abs(grad_b{i})))); 
    
    if temp_W > gradcheck_W
        gradcheck_W = temp_W;
    end
    if temp_b > gradcheck_b
        gradcheck_b = temp_b;
    end

end


end
