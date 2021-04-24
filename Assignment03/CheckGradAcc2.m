function [gradcheck_W, gradcheck_b, gradcheck_gamma, gradcheck_beta] = CheckGradAcc2(ngrads, Grads)
% Input  -  ngrads          (mxn)   nodes x batch size
%        -  Grads           (mxn)   nodes x batch size
% Output -  gradcheck_W     (1x1)    - max error
%        -  gradcheck_b     (1x1)    - max error
% Output -  gradcheck_gamma (1x1)    - max error
%        -  gradcheck_beta  (1x1)    - max error

% Numerical
ngrad_W = ngrads.W;
ngrad_b = ngrads.b;
ngrad_gamma = ngrads.gamma;
ngrad_beta  = ngrads.beta;

% Analytical
grad_W = Grads.W;
grad_b = Grads.b;
grad_gamma = Grads.gamma;
grad_beta  = Grads.beta;

% Maximum error
gradcheck_W = cell(1, numel(ngrad_W));
gradcheck_b = cell(1, numel(ngrad_W));
gradcheck_gamma = cell(1, numel(ngrad_gamma));
gradcheck_beta  = cell(1, numel(ngrad_gamma));

for i = 1 : numel(ngrad_W)
    
    gradcheck_W{i} = max(max(abs(ngrad_W{i} - grad_W{i})./max(0.01, abs(ngrad_W{i}) + abs(grad_W{i}))));
    gradcheck_b{i} = max(max(abs(ngrad_b{i} - grad_b{i})./max(0.01, abs(ngrad_b{i}) + abs(grad_b{i})))); 

end

for i = 1 : numel(ngrad_gamma)
    
    gradcheck_gamma{i} = max(max(abs(ngrad_gamma{i} - grad_gamma{i})./max(0.01, abs(ngrad_gamma{i}) + abs(grad_gamma{i}))));
    gradcheck_beta{i}  = max(max(abs(ngrad_beta{i}  - grad_beta{i}) ./max(0.01, abs(ngrad_beta{i})  + abs(grad_beta{i}))));
     
end

end
