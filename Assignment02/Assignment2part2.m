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

trainX = [X1, X2, X3, X4, X5(:, 1:5000)];
trainY = [Y1, Y2, Y3, Y4, Y5(:, 1:5000)];
trainy = [y1, y2, y3, y4, y5(:, 1:5000)];

validX = X5(:, 5001:10000);
validY = Y5(:, 5001:10000);
validy = y5(:, 5001:10000);

[testX, testY, testy] = LoadBatch("test_batch.mat");

% Data pre-processing - batch normalisation
[trainX, validX, testX] = DataPreProcess(trainX, validX, testX);

% ---------------------------------------------------------------------
% Ex03 - Train your network with cyclical learning rates (lambda = 0)
% ---------------------------------------------------------------------
n = size(trainX, 2);                                 % number of images
% lambda = 1.6424e-6;                                  % regulation
lambda = 1e-5;
% lambda = 0.01;
eta_min = 1e-5;                                      % eta lower boundary
eta_max = 1e-1;                                      % eta upper boundary
batch_size = 100;                                    % Mini-batch size (n_batch)
step_size = 2*floor(n/batch_size);                   % number of iterations in a half cycle (n_s)
cycle_size = 2*step_size;                            % number of iterations in a full cycle 
n_cycles = 4;                                        % number of cycles
n_epochs = n_cycles*cycle_size/(n/batch_size);       % Total times of training

% Mini-batch gradient descent
GDparams = [batch_size, n_epochs];
CLRparams = [eta_min, eta_max, step_size];
% [loss_train, loss_val, J_train, J_val, acc_train, acc_val, n_iteration, Wstar, bstar] = train(trainX, trainY, trainy, validX, validY, validy, GDparams, CLRparams, lambda);

% % Plot - Cost, Loss, Accuracy
% x = 1 : n_iteration;
% figure(1)
% subplot(1,3,1)
% plot(x(1:10:end), J_train(1:10:end), x(1:10:end), J_val(1:10:end))
% title('Cost function')
% xlabel('update steps')
% ylabel('cost')
% legend('training', 'validation')
% 
% subplot(1,3,2)
% plot(x(1:10:end), loss_train(1:10:end), x(1:10:end), loss_val(1:10:end))
% title('Loss function')
% xlabel('update steps')
% ylabel('loss')
% legend('training', 'validation')
% 
% subplot(1,3,3)
% plot(x(1:10:end), acc_train(1:10:end), x(1:10:end), acc_val(1:10:end))
% title('Accuracy')
% xlabel('update steps')
% ylabel('accuracy')
% legend('training', 'validation')

% acc_test = ComputeAccuracy(testX, testy, Wstar, bstar)
% ---------------------------------------------------------------------
% Ex04 - Train your network with CLR & Coarse-to-fine random search lambda
% ---------------------------------------------------------------------
l_min = -4.2;
l_max = -2.5;

% Generate lambda
for i = 1 : 8
    l(i) = l_min + (l_max - l_min)*rand(1, 1);
    lambda(i) = 10^l(i);
end

% Calculate accuracy
for i = 1 : 8
    lambda_acc(i) = CoarseSearch(trainX, trainY, testX, testy, GDparams, CLRparams, lambda(i));
    disp('Done!');
end

result = [l;lambda;lambda_acc]

