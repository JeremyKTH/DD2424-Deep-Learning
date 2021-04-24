function [mu, v] = Mean_Var(s)
% Input  -  s      (mxn)   nodes x batch size
% Output -  mu     (mxn)   nodes x batch size
%        -  v      (mxn)   nodes x batch size

n_b = size(s, 2);          % batch size
mu  = mean(s, 2);          % mean
v   = var(s, 0, 2);
v   = v * (n_b -1) / n_b;  % variance
end