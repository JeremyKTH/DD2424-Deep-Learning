clc;
clear;
addpath C:\Users\jerem\Documents\MATLAB\DeepLearning\Assignment01\Datasets\cifar-10-batches-mat;

% Import data
[trainX, trainY, trainy] = LoadBatch("data_batch_1.mat");
[validX, validY, validy] = LoadBatch("data_batch_2.mat");
[testX, testY, testy] = LoadBatch("test_batch.mat");

% Parameters
K = size(trainY, 1);        % image classes
d = size(trainX, 1);        % image dimensions
W = 0.01.*randn([K d]);     % weight - Gaussian random values, mean = 0, std = 0.01
b = 0.01.*randn([K 1]);     % bias   - Gaussian random values, mean = 0, std = 0.01
try_size = 50;              % small size for testing gradient calculation
lambda = 0.1;               % regulation. When lambda = 0, there is no regularization
n_epochs = 40;              % Total times of training
n_batch = 100;              % Mini-batch size
eta = 0.001;                % learning rate

% Evaluate classifier - return probability
% P = EvaluateClassifier(trainX(:, 1:100), W, b);

% Forward pass - Compute cost function
% J = ComputeCost(trainX(:, 1:100), trainY(:, 1:100), W, b, lambda);

% Compute accuracy
% acc = ComputeAccuracy(trainX(:, 1:100), trainy(1:100), W, b);

% % Compute gradients - Numerical
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1:try_size), trainY(:, 1:try_size), W, b, lambda, 1e-6);
% 
% % Compute gradients - Analytical
% P = EvaluateClassifier(trainX(:, 1:try_size), W, b);
% [grad_W, grad_b] = ComputeGradients(trainX(:, 1:try_size), trainY(:, 1:try_size), P, W, lambda);
% 
% % Check gradients calculation precision
% gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(0, abs(ngrad_W) + abs(grad_W))));
% gradcheck_b = max(max(abs(ngrad_b - grad_b)./max(0, abs(ngrad_b) + abs(grad_b))));

% Mini-batch gradient descent
GDparams = [n_batch, eta, n_epochs];
[Wstar, bstar, J_train, J_valid] = MiniBatchGD(trainX, trainY, validX, validY, GDparams, W, b, lambda);

% Compute training accuracy
acc_train = ComputeAccuracy(trainX, trainy, Wstar, bstar);
acc_valid = ComputeAccuracy(validX, validy, Wstar, bstar);
acc_test = ComputeAccuracy(testX, testy, Wstar, bstar);

% % Cost function chart
% figure('Name', 'Cost function - epoch');
% plot(J_train);
% hold on;
% plot(J_valid);
% grid on;
% legend('training loss', 'validation loss');
% xlabel('epoch');
% ylabel('cost');

% % Visualise W as an image
% for i=1:10
% im = reshape(Wstar(i, :), 32, 32, 3);
% s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
% s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% figure('Name', 'Weight matrix visualisation');
% imshow(cell2mat(s_im));
