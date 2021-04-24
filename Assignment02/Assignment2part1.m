clc; close all; clear;
addpath C:\Users\jerem\Documents\MATLAB\DeepLearning\Assignment02\Datasets\cifar-10-batches-mat;
% ---------------------------------------------------------------------
% Ex01 - Read in the data & initialize the parameters of the network
% ---------------------------------------------------------------------
% Import data
[X1, Y1, y1] = LoadBatch("data_batch_1.mat");
[X2, Y2, y2] = LoadBatch("data_batch_2.mat");
[X3, Y3, y3] = LoadBatch("data_batch_3.mat");
[X4, Y4, y4] = LoadBatch("data_batch_4.mat");
[X5, Y5, y5] = LoadBatch("data_batch_5.mat");

trainX = X1;
trainY = Y1;
trainy = y1;

validX = X5(:, 5001:10000);
validY = Y5(:, 5001:10000);
validy = y5(:, 5001:10000);

[testX, testY, testy] = LoadBatch("test_batch.mat");

% Data pre-processing
[trainX, validX, testX] = DataPreProcess(trainX, validX, testX);

% Parameters
[W, b] = init_param();  % initialise weight, bias
 
% ---------------------------------------------------------------------
% Ex02 - Compute the gradients for the network parameters
% ---------------------------------------------------------------------
try_size = 100;          % small size for testing gradient calculation
lambda = 0;             % regulation

% Compute gradients - Numerical
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1:try_size), trainY(:, 1:try_size), W, b, lambda, 1e-6);

% Compute gradients - Analytical
[grad_W, grad_b] = ComputeGradients(trainX(:, 1:try_size), trainY(:, 1:try_size), W, b, lambda);

% Check gradients calculation precision
gradcheck_W1 = max(max(abs(ngrad_W{1} - grad_W{1})./max(0.01, abs(ngrad_W{1}) + abs(grad_W{1}))));
gradcheck_b1 = max(max(abs(ngrad_b{1} - grad_b{1})./max(0.01, abs(ngrad_b{1}) + abs(grad_b{1}))));
gradcheck_W2 = max(max(abs(ngrad_W{2} - grad_W{2})./max(0, abs(ngrad_W{2}) + abs(grad_W{2}))));
gradcheck_b2 = max(max(abs(ngrad_b{2} - grad_b{2})./max(0, abs(ngrad_b{2}) + abs(grad_b{2}))));

% ---------------------------------------------------------------------
% Ex03 - Train your network with cyclical learning rates (lambda = 0)
% ---------------------------------------------------------------------
% lambda = 0;                             % regulation
% eta_min = 1e-5;                         % eta lower boundary
% eta_max = 1e-1;                         % eta upper boundary
% step_size = 500;                        % number of iterations in a half cycle
% cycle_size = 2*step_size;               % number of iterations in a full cycle 
% n_cycles = 1;                           % number of cycles
% batch_size = 100;                       % Mini-batch size
% n_epochs = n_cycles*cycle_size/(size(trainX, 2)/batch_size);  % Total times of training
% 
% % Mini-batch gradient descent
% GDparams = [batch_size, n_epochs];
% CLRparams = [eta_min, eta_max, step_size];
% [Wstar, bstar, J_train, J_valid] = MiniBatchGD(trainX, trainY, validX, validY, GDparams, CLRparams, W, b, lambda);
% 
% % Compute training accuracy
% acc_train = ComputeAccuracy(trainX, trainy, Wstar, bstar);
% acc_valid = ComputeAccuracy(validX, validy, Wstar, bstar);
% acc_test = ComputeAccuracy(testX, testy, Wstar, bstar);
% 
% % Cost function chart
% figure('Name', 'Cost function - update steps');
% plot(J_train);
% hold on;
% plot(J_valid);
% grid on;
% legend('training loss', 'validation loss');
% xlabel('update steps');
% ylabel('cost');

% ---------------------------------------------------------------------
% Ex04 - Train your network with CLR & Coarse-to-fine random search lambda
% ---------------------------------------------------------------------

% % Visualise W as an image
% for i=1:10
% im = reshape(Wstar(i, :), 32, 32, 3);
% s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
% s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% figure('Name', 'Weight matrix visualisation');
% imshow(cell2mat(s_im));
