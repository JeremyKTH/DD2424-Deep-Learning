function [X, Y, y] = LoadBatch(filename)
% Input  -  filename
% Output -  X  (dxn)
%        -  Y  (Kxn)
%        -  y  (1xn)
%
% Note:
% d : dimension of each image (3072)
% n : number of images (10,000)
% K : number of classes (10)

data = load(filename);
X = double(data.data')/double(255); % convert image from 0-256 to 0-1
y = double(data.labels') + 1;

% One-hot encoding
K = length(unique(y)); % unique returns data in y without repetition
n = size(X, 2);
Y = zeros(K, n);

for i = 1 : n
    Y(y(i), i) = 1;
end

end