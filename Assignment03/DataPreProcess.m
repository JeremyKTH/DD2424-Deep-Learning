function [trainX, validX, testX] = DataPreProcess(trainX, validX, testX)
% Input  -  trainX       (dxn)
%        -  validX       (dx5000)
%        -  testX        (dxn)
% Output -  trainX       (dxn)
%        -  validX       (dx5000)
%        -  testX        (dxn)

mean_X = mean(trainX, 2);
std_X = std(trainX, 0, 2);

trainX = trainX - repmat(mean_X, [1, size(trainX, 2)]);
trainX = trainX ./ repmat(std_X, [1, size(trainX, 2)]);

validX = validX - repmat(mean_X, [1, size(validX, 2)]);
validX = validX ./ repmat(std_X, [1, size(validX, 2)]);

testX = testX - repmat(mean_X, [1, size(testX, 2)]);
testX = testX ./ repmat(std_X, [1, size(testX, 2)]);
end
