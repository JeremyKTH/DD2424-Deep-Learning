function Grads = ComputeGradients(X, Y, NetParams, lambda, train)
% Input  -  X           (dxn)
%        -  Y           (Kxn)
%        -  NetParams   (struct) 
%        -  lambda      (1x1)
%        -  train       (1x1)
% Output -  grad_W      (1xL)    - L : num of layer
%        -  grad_b      (1xL)    - L : num of layer

% Variables
n_b         = size(X,2);     % batch size
W           = NetParams.W;
numOfLayers = numel(W);

% Initialisation
grad_W     = cell(1,numOfLayers);
grad_b     = cell(1,numOfLayers);
grad_gamma = cell(1,numOfLayers-1);
grad_beta  = cell(1,numOfLayers-1);

% ----- Forward Pass ----- %
ForwardPass = EvaluateClassifier(X, NetParams, train);
P = ForwardPass.P;  % Probability
H = ForwardPass.h;  % reLu activation value

% ----- Backward Pass ----- %
% Propagate the gradient through the loss and softmax operation
G_b = -(Y-P);

if NetParams.use_bn
% ----- with Batch Normalisation ----- %
    % Propagate the gradient trhough the loss and softmax operation
    grad_W{numOfLayers} = (1/n_b)*(G_b*H{numOfLayers}') + 2*lambda*W{numOfLayers};
    grad_b{numOfLayers} = (1/n_b)*(G_b*ones(n_b,1));
    
    % hidden -> Input layer
    G_b = W{numOfLayers}'*G_b;
    G_b = G_b.*sign(H{numOfLayers}); % reLu
    
    mu    = ForwardPass.mu;
    v     = ForwardPass.v;
    s     = ForwardPass.s;
    s_hat = ForwardPass.s_hat;
    gamma = NetParams.gamma;
    
    for i = (numOfLayers-1) : -1 : 1
        % Compute gradient for the scale and offset parameters for layer i
        grad_gamma{i} = (1/n_b)*(G_b.*s_hat{i})*ones(n_b, 1);
        grad_beta{i}  = (1/n_b)*(G_b*ones(n_b, 1));

        % Propagate the gradients through the scale and shift
        G_b = G_b.*(gamma{i}*ones(1, n_b));
        
        % Propagate Gbatch through the batch normalization
        G_b = BatchNormBackPass(G_b, s{i}, mu{i}, v{i});
        
        % The gradients of J w.r.t. bias vector b{i} and W{i}
        grad_W{i} = (1/n_b)*(G_b*H{i}') + 2*lambda*W{i};
        grad_b{i} = (1/n_b)*G_b*ones(n_b, 1);
        
        % If i > 1, propagate G_b to the previous layer
        if i > 1
            G_b = W{i}'*G_b;
            G_b = G_b.*sign(H{i});
        end
    end
    Grads.gamma = grad_gamma;
    Grads.beta  = grad_beta;
    Grads.mu    = mu;
    Grads.v     = v;

else
% ----- without Batch Normalisation ----- %
    for i = numOfLayers : -1 : 1
        grad_W{i} = (1/n_b)*(G_b*H{i}') + 2*lambda*W{i};
        grad_b{i} = (1/n_b)*G_b*ones(n_b, 1);
        % hidden -> Input layer
        G_b = W{i}'*G_b;
        G_b = G_b.*sign(H{i}); % reLu
    end
end

Grads.W     = grad_W;
Grads.b     = grad_b;

end
