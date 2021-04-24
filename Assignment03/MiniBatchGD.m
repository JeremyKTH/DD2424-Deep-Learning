function NetParams = MiniBatchGD(Xbatch, Ybatch, NetParams, eta, lambda, alpha, train)
% Input  -  Xbatch    (dxn) - for training
%        -  Ybatch    (1xn)
%        -  NetParams (struct) 
%        -  eta       (1x1)
%        -  lambda    (1x1)
%        -  train     (1x1)
% Output -  NetParams (struct) 

% Compute gradients
Grads = ComputeGradients(Xbatch, Ybatch, NetParams, lambda, train);

% Initialisation
W = NetParams.W;
b = NetParams.b;
grad_W = Grads.W;
grad_b = Grads.b;
Wstar = cell(1,numel(W));
bstar = cell(1,numel(W));

% Update weights & biases
for i = 1 : numel(W)
    Wstar{i} = W{i} - eta*grad_W{i};
    bstar{i} = b{i} - eta*grad_b{i};
end

NetParams.W = Wstar;
NetParams.b = bstar;

if NetParams.use_bn
    grad_gamma = Grads.gamma;
    grad_beta  = Grads.beta;
    gamma = NetParams.gamma;
    beta  = NetParams.beta;
    gamma_star = cell(1,numel(W)-1);
    beta_star  = cell(1,numel(W)-1);
    
    % If mu & v is already calculated
    if numel(fieldnames(NetParams)) == 7
        mu_avg = NetParams.mu_avg;
        v_avg  = NetParams.v_avg;
        mu     = Grads.mu;
        v      = Grads.v;
        
        % Exponential moving avg for batch means and variances
        for i = 1 : (numel(W) - 1)
            mu_avg{i} = alpha*mu_avg{i} + (1-alpha)*mu{i};
            v_avg{i}  = alpha*v_avg{i}  + (1-alpha)*v{i};
        end
    else
        mu_avg = Grads.mu;
        v_avg  = Grads.v;

    end

    NetParams.mu_avg = mu_avg;
    NetParams.v_avg  = v_avg;
    
    % Update gamma & beta ( scale & shift )
    for i = 1 : numel(gamma)
        gamma_star{i} = gamma{i} - eta*grad_gamma{i};
        beta_star{i}  = beta{i}  - eta*grad_beta{i};
    end
    NetParams.gamma = gamma_star;
    NetParams.beta  = beta_star;   % Check NetParams, Grads
end
end
