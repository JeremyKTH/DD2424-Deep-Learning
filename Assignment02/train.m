function [loss_train, loss_val, J_train, J_val, acc_train, acc_val, n_iteration, Wstar, bstar] = train(X, Y, y, val_X, val_Y, val_y, GDparams, CLRparams, lambda)
% Input  -  X            (dxn) - for training
%        -  y            (1xn)
%        -  val_X        (dxn) - for validation
%        -  val_Y        (1xn)
%        -  val_y        (1xn)
%        -  GDparams     (1x2) - [batch_size, n_epoch]
%        -  CLRparams    (1x3) - [eta_min, eta_max, step_size]
%        -  lambda       (1x1)
% Output -  loss_train   (1x1)
%        -  loss_val     (1x1)
%        -  J_train      (1x1)
%        -  J_val        (1x1)
%        -  acc_train    (1x1)
%        -  acc_val      (1x1)
%        -  n_iteration  (1x1) - total num of iteration
%        -  W*           (Kxd)
%        -  b*           (Kx1)

% initialise weight, bias
[Wstar, bstar] = init_param();  

% Mini batch params
batch_size = GDparams(1);
n_epochs = GDparams(2);

% Loss
loss_train = zeros(size(n_epochs));
loss_val = zeros(size(n_epochs));
% Cost
J_train = zeros(size(n_epochs));
J_val = zeros(size(n_epochs));

% Accuracy
acc_train = zeros(size(n_epochs));
acc_val = zeros(size(n_epochs));

% iteration per epoch
iter = size(X, 2)/batch_size;


for i = 1 : n_epochs % for each epoch i
    for j = 1 : ((size(X, 2))/batch_size) % for each iteration j
        % Run thorugh images sequentially
        j_start = (j-1)*batch_size + 1;
        j_end = j*batch_size;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        ybatch = y(j_start:j_end);
        
        % Cyclical Learning Rate
        n_iteration = (i-1)*iter + j;
        eta = CyclicalLearningRate(CLRparams, n_iteration);
        
        % Mini-batch gradient descent
        [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch, Wstar, bstar, eta, lambda);

        % Every 10 iteration
        if mod(n_iteration, 10) == 1
            % Compute loss, cost, accuracy - training data
            [cost, loss] = ComputeCost(Xbatch, Ybatch, Wstar, bstar, lambda);
            J_train(n_iteration) = cost;
            loss_train(n_iteration) = loss;
            acc_train(n_iteration) = ComputeAccuracy(Xbatch, ybatch, Wstar, bstar);
            
            % Compute loss, cost, accuracy - validation data
            [cost, loss] = ComputeCost(val_X, val_Y, Wstar, bstar, lambda);
            J_val(n_iteration) = cost;
            loss_val(n_iteration) = loss;
            acc_val(n_iteration) = ComputeAccuracy(val_X, val_y, Wstar, bstar);
        end
    end    
end

end
