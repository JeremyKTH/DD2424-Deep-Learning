function G_b = BatchNormBackPass(G_b, s, mu, v)
% Input  -  G_b  (mxn)   nodes x batch size
%        -  s    (mxn)   nodes x batch size
%        -  mu   (mxn)   num of layers -1
%        -  v    (mxn)   num of layers -1
% Output -  G_b  (mxn)   nodes x batch size
    n = size(s,2); 
    
    sigma1 = ((v+eps).^-0.5)';
    sigma2 = ((v+eps).^-1.5)';
    
    G1 = G_b.*(sigma1'*ones(1,n));
    G2 = G_b.*(sigma2'*ones(1,n));
    
    D = s - mu*ones(1,n);
    c = (G2.*D)*ones(n,1);
    
    G_b = G1 - (1/n)*(G1*ones(n,1)*ones(1,n)) - (1/n)*D.*(c*ones(1,n));
end