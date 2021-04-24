function [Wstar, bstar, J, J_val] = MiniBatchGD(X, Y, val_X, val_Y, GDparams, W, b, lambda)
% Input  -  X         (dxn) - for training
%        -  y         (1xn)
%        -  val_X     (dxn) - for validation
%        -  val_Y     (1xn)
%        -  GDparams  (1x3) - [n_batch, eta, n_epoch]
%        -  W         (Kxd)
%        -  b         (Kx1)
%        -  lambda    (1x1)
% Output -  W*        (Kxd)
%        -  b*        (Kx1)
%        -  J         (1x1)
%        -  J_val     (1x1)

n_batch = GDparams(1);
eta = GDparams(2);
n_epoch = GDparams(3);
J = zeros(size(n_epoch));
J_val = zeros(size(n_epoch));

for i = 1 : n_epoch % for each epoch
    for j = 1 : ((size(X, 2))/n_batch) % for each mini-batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        % Compute gradients
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        
        % Update weight & bias
        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
    % Compute costs
    J(i) = ComputeCost(X, Y, W, b, lambda);
    J_val(i) = ComputeCost(val_X, val_Y, W, b, lambda);
        
end
Wstar = W;
bstar = b;
end
