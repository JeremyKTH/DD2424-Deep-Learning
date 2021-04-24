function [s_hat] = BatchNormalise(s, mu, v)
% Input  -  s      (mxn)   nodes x batch size
%        -  mu     (mxn)   num of layers -1
%        -  v      (mxn)   num of layers -1
% Output -  s_hat  (mxn)   nodes x batch size

% Batch Normalisation
s_hat = (diag((v+eps).^-0.5))*(s-mu); 
end