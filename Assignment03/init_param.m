function NetParams = init_param(layer, use_bn)
% Input  -  layer      (1xL)     - L = number of layers
%        -  use_bn     (binary)  - 0 = no BN; 1 = use BN 
% Output -  NetParams  (struct) 

len = length(layer);
NetParams.use_bn = use_bn;

% Only layers with learnable weights are considered the depth of NN
% in other words, all layers except input layer count as layers in a k-layers NN

if use_bn
% ----- with Batch Normalisation ----- %
    for i = 2 : (len-1)
    NetParams.gamma{i-1} = ones(layer(i), 1);
    NetParams.beta {i-1} = zeros(layer(i), 1);
    end
end

for i = 2 : len
    % He initialisation
    NetParams.W{i-1} = sqrt(2/(layer(i-1))).*randn(layer(i), layer(i-1));
    NetParams.b{i-1} = zeros(layer(i), 1);

    % Sigma initialisation
%     sig = 1e-4;
%     NetParams.W{i-1} = sig.*randn(layer(i), layer(i-1));
%     NetParams.b{i-1} = zeros(layer(i), 1);

end
end
