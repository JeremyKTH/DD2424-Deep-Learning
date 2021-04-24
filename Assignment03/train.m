function [loss_train, loss_val, J_train, J_val, acc_train, acc_val, n_iteration, NetParams] = train(X, Y, y, val_X, val_Y, val_y, GDparams, CLRparams, lambda, layer, use_bn, alpha, shuffle)
% Input  -  X            (dxn)    - for training
%        -  y            (1xn)
%        -  val_X        (dxn)    - for validation
%        -  val_Y        (1xn)
%        -  val_y        (1xn)
%        -  GDparams     (1x2)    - [batch_size, n_epoch]
%        -  CLRparams    (1x3)    - [eta_min, eta_max, step_size]
%        -  lambda       (1x1)
%        -  layer        (1xL)    - nuerons in each layer
%        -  use_bn       (binary) - w or w/o batch normalisation
%        -  alpha        (1x1)    - exponential moving avg
%        -  shuffle      (1x1)    - shuffle data after every epoch
% Output -  loss_train   (1x1)
%        -  loss_val     (1x1)
%        -  J_train      (1x1)
%        -  J_val        (1x1)
%        -  acc_train    (1x1)
%        -  acc_val      (1x1)
%        -  n_iteration  (1x1)    - total num of iteration
%        -  NetParams   (struct)

% initialise weight, bias
NetParams = init_param(layer, use_bn);

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
        if shuffle
%             j_start = (j-1)*batch_size + 1;
%             j_end = j*batch_size;
            Batch_indizies = randperm((size(X, 2)), batch_size);
            Xbatch = X(:,Batch_indizies);
            Ybatch = Y(:,Batch_indizies);
            ybatch = y(:,Batch_indizies);
%             ybatch = y(j_start:j_end);
        else
            % Run thorugh images sequentially
            j_start = (j-1)*batch_size + 1;
            j_end = j*batch_size;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            ybatch = y(j_start:j_end);
        end

          % Cyclical Learning Rate
          n_iteration = (i-1)*iter + j;
          eta = CyclicalLearningRate(CLRparams, n_iteration);

          % indicate training
          train = 1;

          % Mini-batch gradient descent
          NetParams = MiniBatchGD(Xbatch, Ybatch, NetParams, eta, lambda, alpha, train);
          
% --------------------------------------------------------------- %
          if NetParams.use_bn   % with Batch Normalisation
% --------------------------------------------------------------- %
              % not training
              train = 0;
              % Every 10 iteration
              if mod(n_iteration, 100) == 1
                  % Compute loss, cost, accuracy - training data
                  [cost, loss] = ComputeCost(Xbatch, Ybatch, NetParams, lambda, train);
                  acc_train(n_iteration) = ComputeAccuracy(Xbatch, ybatch, NetParams, train);
                  J_train(n_iteration) = cost;
                  loss_train(n_iteration) = loss;

                  % Compute loss, cost, accuracy - validation data
                  [cost, loss] = ComputeCost(val_X, val_Y, NetParams, lambda, train);
                  acc_val(n_iteration) = ComputeAccuracy(val_X, val_y, NetParams, train);
                  J_val(n_iteration) = cost;
                  loss_val(n_iteration) = loss;
              end
% --------------------------------------------------------------- %
          else                 % without Batch Normalisation
% --------------------------------------------------------------- %
              % Every 10 iteration
              if mod(n_iteration, 50) == 1
                  train = 0;
                  
                  % Compute loss, cost, accuracy - training data
                  [cost, loss] = ComputeCost(Xbatch, Ybatch, NetParams, lambda, train);
                  J_train(n_iteration) = cost;
                  loss_train(n_iteration) = loss;
                  acc_train(n_iteration) = ComputeAccuracy(Xbatch, ybatch, NetParams, train);

                  % Compute loss, cost, accuracy - validation data
                  [cost, loss] = ComputeCost(val_X, val_Y, NetParams, lambda, train);
                  J_val(n_iteration) = cost;
                  loss_val(n_iteration) = loss;
                  acc_val(n_iteration) = ComputeAccuracy(val_X, val_y, NetParams, train);
              end
          end
    end    
end

end
