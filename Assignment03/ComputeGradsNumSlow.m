function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gamma = cell(numel(NetParams.gamma), 1);
    Grads.beta  = cell(numel(NetParams.beta), 1);
end

train = 1;

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
%         c1 = ComputeCost(X, Y, NetTry, lambda);        
        [c1, ~] = ComputeCost(X, Y, NetTry, lambda, train);   
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
%         c2 = ComputeCost(X, Y, NetTry, lambda);
        [c2, ~] = ComputeCost(X, Y, NetTry, lambda, train);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
%         c1 = ComputeCost(X, Y, NetTry, lambda);
        [c1, ~] = ComputeCost(X, Y, NetTry, lambda, train);
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
%         c2 = ComputeCost(X, Y, NetTry, lambda);
        [c2, ~] = ComputeCost(X, Y, NetTry, lambda, train);
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gamma)
        Grads.gamma{j} = zeros(size(NetParams.gamma{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gamma{j})
            
            gammas_try = NetParams.gamma;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gamma = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda, train);
            
            gammas_try = NetParams.gamma;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gamma = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda, train);
            
            Grads.gamma{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.beta)
        Grads.beta{j} = zeros(size(NetParams.beta{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.beta{j})
            
            betas_try = NetParams.beta;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.beta = betas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda, train);
            
            betas_try = NetParams.beta;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.beta = betas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda, train);
            
            Grads.beta{j}(i) = (c2-c1) / (2*h);
        end
    end    
end

end