function acc = CoarseSearch(X, Y, testX, testy, GDparams, CLRparams, lambda)
% Input  -  X            (dxn) - for training
%        -  y            (1xn)
%        -  testX        (dxn) - for testing
%        -  testY        (1xn)
%        -  GDparams     (1x2) - [batch_size, n_epoch]
%        -  CLRparams    (1x3) - [eta_min, eta_max, step_size]
%        -  lambda       (1x1)
% Output -  acc  (1x1) - accuracy of a specific lambda

% initialise weight, bias
[Wstar, bstar] = init_param();  

% Mini batch params
batch_size = GDparams(1);
n_epochs = GDparams(2);

% iteration per epoch
iter = size(X, 2)/batch_size;

for i = 1 : n_epochs % for each epoch i
    for j = 1 : ((size(X, 2))/batch_size) % for each iteration j
        % Run thorugh images sequentially
        j_start = (j-1)*batch_size + 1;
        j_end = j*batch_size;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        
        % Cyclical Learning Rate
        n_iteration = (i-1)*iter + j;
        eta = CyclicalLearningRate(CLRparams, n_iteration);
        
        % Mini-batch gradient descent
        [Wstar, bstar] = MiniBatchGD(Xbatch, Ybatch, Wstar, bstar, eta, lambda);
    end    
end

% Test network accuracy after 1 cycle
acc = ComputeAccuracy(testX, testy, Wstar, bstar);
end
