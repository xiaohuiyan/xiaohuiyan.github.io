function [U, cur_error] = snmf_als(S, K, lambda)
% @function
%   Using alternating least squares method to solve ridge regression of 
%   matrix factorization problem: S = UU' + E
% $ Arguments $
%   - S   M*M term correlation matrix
%   - K   topic number
%   - lambda   regularize coefficient for U
% $ Return $
%   - U     M*K term-topic matrix
% $ Author $
%   Xiaohui Yan(xhcloud@gmail.com), 2011-10-30
    
if nargin == 2
    lambda = 0;
end

M = size(S, 1);
U = rand(M, K);

eps = 10e-9;
cur_error = sum(sum((S - U * U').^2));
errs = cur_error;
fprintf('Random init error = %.6f \n', cur_error);

max_iter = 300;
alpha = 0.5;   % U_new = alpha * cur_U + (1-alpha) * last_U
iter = 0;

imp = 1;  %improvement
thres_imp = 1e-8;  % stop iteration when relative improvement less than it

% ALS not always decrease error
while imp > thres_imp && iter < max_iter
    iter = iter + 1;
    
    % update T,D by ALS algorithm
    cur_U = S * U * pinv(U'* U + lambda * eye(K));
    % filter negative elements
    cur_U = bsxfun(@max, cur_U, eps);
    % smoothing, important for convergence
    cur_U = alpha * cur_U + (1-alpha) * U;  
    
    % compute error    
    err = sum(sum((S - cur_U * cur_U').^2));
    errU = lambda * norm(cur_U, 'fro')^2;
    cur_error = err + errU;
    errs = [errs cur_error];
    
    imp = abs(errs(end-1) - errs(end))/errs(1);
    fprintf('iter %d error=%e, residual=%e, errU=%e, imp=%e\n', ...
        iter, cur_error, err, errU, imp);
    
    if cur_error > errs(end-1)
        fprintf('cur_error is worse.\n');
        if iter > 30
            return
        end
    end
    
    U = cur_U;
end

U(U<=eps) = 0;
return