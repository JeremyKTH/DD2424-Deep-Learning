function ForwardPass = EvaluateClassifier(X, NetParams, train)
% Input  -  X           (dxn)
%        -  NetParams   (struct)
%        -  train       (1x1)
% Output -  P           (Kxn)
%        -  h           (Kxn)

n_b = size(X,2);  % batch size
W   = NetParams.W;
b   = NetParams.b;
numOfLayers = numel(W);

% s_tilde = cell(1, numel(W)-1);
s_hat = cell(1, numOfLayers-1);
s     = cell(1, numOfLayers-1);    % minus 1 because s_out is 
mu    = cell(1, numOfLayers-1);
v     = cell(1, numOfLayers-1);
h     = cell(1, numOfLayers);
h{1}  = X;


if NetParams.use_bn
% ----- with Batch Normalisation ----- %
    for i = 1 : (numOfLayers - 1)
        % linear transformation
        s{i} = W{i}*h{i} + b{i}*ones(1, n_b);
    
        if train == 0
            mu = NetParams.mu_avg;
            v  = NetParams.v_avg;
        else
            [mu{i}, v{i}] = Mean_Var(s{i});
        end
        
%         [mu{i}, v{i}] = Mean_Var(s{i});
        
        % Batch normalisation
        s_hat{i} = BatchNormalise(s{i}, mu{i}, v{i});
                
        % Rescale and shift
        gamma = NetParams.gamma{i};
        beta  = NetParams.beta{i};
        s_tilde = gamma.*s_hat{i} + beta;
%         disp(s_tilde);
        
        % reLu
%         h{i+1} = max(0.01*s_tilde, s_tilde);
        h{i+1} = max(0, s_tilde);
          
    end
else
% ----- without Batch Normalisation ----- %
    for i = 1 : (numOfLayers - 1)
        % linear transformation
        s{i} = W{i}*h{i} + b{i}*ones(1, n_b);
        
        % reLu
        h{i+1} = max(s{i}, 0);
    end
end


% Output layer
s_out = W{numOfLayers}*h{numOfLayers} + b{numOfLayers}*ones(1, n_b);
P = softmax(s_out);

ForwardPass.h     = h;      % reLu
ForwardPass.P     = P;      % probability
ForwardPass.mu    = mu;     % mean of score in each layer
ForwardPass.v     = v;      % variance of score in each layer
ForwardPass.s     = s;      % scores w/o Batch Normalisation
ForwardPass.s_hat = s_hat;  % scores with Batch Normalisation
ForwardPass.s_out = s_out;

end
