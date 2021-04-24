function acc = CoarseSearch(X, Y, testX, testy, GDparams, CLRparams, lambda, layer, use_bn, alpha, shuffle)
% Input  -  X            (dxn) - for training
%        -  y            (1xn)
%        -  testX        (dxn) - for testing
%        -  testY        (1xn)
%        -  GDparams     (1x2) - [batch_size, n_epoch]
%        -  CLRparams    (1x3) - [eta_min, eta_max, step_size]
%        -  lambda       (1x1)
%        -  layer        (1xL)
%        -  use_bn       (binary)
%        -  alpha        (1x1) - exponential moving avg
%        -  shuffle      (1x1) - shuffle training batch every epoch
% Output -  acc  (1x1) - accuracy of a specific lambda

% initialise weight, bias
NetParams = init_param(layer, use_bn);

% Mini batch params
batch_size = GDparams(1);
n_epochs = GDparams(2);

% iteration per epoch
iter = size(X, 2)/batch_size;

for i = 1 : n_epochs % for each epoch i
    for j = 1 : ((size(X, 2))/batch_size) % for each iteration j
        
        if shuffle
            Batch_indizies = randperm((size(X, 2)), batch_size);
            Xbatch = X(:,Batch_indizies);
            Ybatch = Y(:,Batch_indizies);
        else
            % Run thorugh images sequentially
            j_start = (j-1)*batch_size + 1;
            j_end = j*batch_size;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
        end
        
        % Cyclical Learning Rate
        n_iteration = (i-1)*iter + j;
        eta = CyclicalLearningRate(CLRparams, n_iteration);
        
        % indicate training
        train = 1;
        
        % Mini-batch gradient descent
        NetParams = MiniBatchGD(Xbatch, Ybatch, NetParams, eta, lambda, alpha, train);
    end    
end

% Test network accuracy after 1 cycle
train = 0;
acc = ComputeAccuracy(testX, testy, NetParams, train);
end
