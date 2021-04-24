function eta = CyclicalLearningRate(CLRparams, n_iter)
% Input  -  CLRparams       (1x3)  - [eta_min, eta_max, step_size]
%        -  n_iter          (1x1)  - nth of iteration
% Output -  eta             (1x1)

eta_min = CLRparams(1);
eta_max = CLRparams(2);
step_size = CLRparams(3);

% Create boundary for cyclical learning rate
n_cycle = floor(n_iter / (2*step_size));
cycle_start = 2*n_cycle*step_size;
cycle_mid = (2*n_cycle+1)*step_size;
cycle_end = 2*(n_cycle+1)*step_size;

% Left half cycle
if (cycle_start <= n_iter) && (n_iter <= cycle_mid)
    eta = eta_min + ((n_iter - 2*n_cycle*step_size)/step_size)*(eta_max-eta_min);
end

% Right half cycle
if (cycle_mid < n_iter) && (n_iter <= cycle_end)
    eta = eta_max - ((n_iter - (2*n_cycle+1)*step_size)/(step_size))*(eta_max-eta_min);
end
end
