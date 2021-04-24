clc; close all; clear;
addpath C:\Users\jerem\Documents\MATLAB\DeepLearning\Assignment03\Datasets\cifar-10-batches-mat;

% Import data
[X1, Y1, y1] = LoadBatch("data_batch_1.mat");
[X2, Y2, y2] = LoadBatch("data_batch_2.mat");
[X3, Y3, y3] = LoadBatch("data_batch_3.mat");
[X4, Y4, y4] = LoadBatch("data_batch_4.mat");
[X5, Y5, y5] = LoadBatch("data_batch_5.mat");

% trainX = X1; trainY = Y1; trainy = y1;

trainX = [X1, X2, X3, X4, X5(:, 1:5000)];
trainY = [Y1, Y2, Y3, Y4, Y5(:, 1:5000)];
trainy = [y1, y2, y3, y4, y5(:, 1:5000)];

validX = X5(:, 5001:10000);
validY = Y5(:, 5001:10000);
validy = y5(:, 5001:10000);

[testX, testY, testy] = LoadBatch("test_batch.mat");

% Data pre-processing - Normalisation
[trainX, validX, testX] = DataPreProcess(trainX, validX, testX);

% ---------------------------------------------------------------------
% Ex01 - K-layer neural network
% ---------------------------------------------------------------------
% Parameters
% use_bn          = 1;      % 0 = no BN; 1 = use BN (BN: Batch Normalisation)
% lambda          = 0;      % regularisation
% try_size        = 50;     % small size for debugging
% numOfClasses    = 10; 
% inputDimensions = 3072;
% layer = [inputDimensions, 50, 50, numOfClasses];   % neurons in each layer

% Initialisation - weight, bias
% NetParams = init_param(layer, use_bn);

% Compute gradients - Numerical
% ngrads = ComputeGradsNumSlow(trainX(:, 1:try_size), trainY(:, 1:try_size), NetParams, lambda, 1e-6);

% Compute gradients - Analytical
% train = 1;
% Grads = ComputeGradients(trainX(:, 1:try_size), trainY(:, 1:try_size), NetParams, lambda, train);

% Check gradients calculation precision
% Without BN
% [gradcheck_W, gradcheck_b] = CheckGradAcc(ngrads, Grads)

% With BN
% [gradcheck_W, gradcheck_b, gradcheck_gamma, gradcheck_beta] = CheckGradAcc2(ngrads, Grads)

% ---------------------------------------------------------------------
% Ex03 - Train your network with cyclical learning rates (lambda = 0)
% ---------------------------------------------------------------------
% Parameters - cyclical learning rate
n = size(trainX, 2);                                 % number of images
eta_min = 1e-5;                                      % eta lower boundary
eta_max = 1e-1;                                      % eta upper boundary
batch_size = 100;                                    % Mini-batch size (n_batch)
% step_size = 2*floor(n/batch_size);                   % number of iterations in a half cycle (n_s)
step_size = 3*floor(n/batch_size);   
cycle_size = 2*step_size;                            % number of iterations in a full cycle 
n_cycles = 5;                                        % number of cycles
n_epochs = n_cycles*cycle_size/(n/batch_size);       % Total times of training
% lambda = 0.005;
% lambda = 10^(-2.4402);
% lambda = 10^(-2.205);
lambda = 10^(-2.4597);

% Parameters - batch normalisation
use_bn = 0;                                          % 0 = no BN; 1 = use BN (BN: Batch Normalisation)
numOfClasses = 10; 
inputDimensions = 3072;
% layer = [inputDimensions, 50, 30, 20, 20, 10, 10, 10, 10, numOfClasses]; % neurons in each layer
layer = [inputDimensions, 50, 50, numOfClasses]; % neurons in each layer
alpha = 0.9;
shuffle = 0;          % shuffle datasets

% Mini-batch gradient descent
GDparams = [batch_size, n_epochs];
CLRparams = [eta_min, eta_max, step_size];
[loss_train, loss_val, J_train, J_val, acc_train, acc_val, n_iteration, NetParams] = train(trainX, trainY, trainy, validX, validY, validy, GDparams, CLRparams, lambda, layer, use_bn, alpha, shuffle);

% Accuracy test
train = 0;
acc_test = ComputeAccuracy(testX, testy, NetParams, train)
% ---------------------------------------------------------------------
% Plot - Cost, Loss, Accuracy
% ---------------------------------------------------------------------
x = 1 : n_iteration;

f = 100; % frequency

figure(1)
subplot(1,3,1)
plot(x(1:f:end), J_train(1:f:end), x(1:f:end), J_val(1:f:end))
title('Cost function')
xlabel('update steps')
ylabel('cost')
legend('training', 'validation')

subplot(1,3,2)
plot(x(1:f:end), loss_train(1:f:end), x(1:f:end), loss_val(1:f:end))
title('Loss function')
xlabel('update steps')
ylabel('loss')
legend('training', 'validation')

subplot(1,3,3)
plot(x(1:f:end), acc_train(1:f:end), x(1:f:end), acc_val(1:f:end))
title('Accuracy')
xlabel('update steps')
ylabel('accuracy')
legend('training', 'validation')

% ---------------------------------------------------------------------
% Ex04 - Train your network with CLR & Coarse-to-fine random search lambda
% ---------------------------------------------------------------------
% l_min = -5;
% l_max = -2.5;
% 
% % Generate lambda
% for i = 1 : 5
%     l(i) = l_min + (l_max - l_min)*rand(1, 1);
%     lambda(i) = 10^l(i);
% end
% 
% % Coarse Search & test model accuracy
% for i = 1 : 5
%     lambda_acc(i) = CoarseSearch(trainX, trainY, testX, testy, GDparams, CLRparams, lambda(i), layer, use_bn, alpha, shuffle);
%     disp('Done!');
% end
% 
% result = [l; lambda; lambda_acc]
