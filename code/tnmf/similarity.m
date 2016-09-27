function [S] = similarity(X, measure)
% compuate similarity matrix
% parameters
%    X    M*N term-doc count matrix
%    measure   'cos' (default) or 'PMI' stands for positive PMI
% return   
%    S   M*M similarity matrix
% author
%    xhcloud@gmail.com, 2012-05-18

if nargin == 1
    measure = 'cos';
end

if strcmp(measure, 'cos')==1
    % compute words similarity matrix by cosine
    nX = normr(X);   % normalize each row with length 1
    S = nX*nX';
elseif strcmp(measure, 'PMI')==1
    eps = 10e-9;
    Cij = X*X' + eps;      % word co-occurrence matrix
    Ci = sum(Cij, 1);
    nAll = sum(Ci);
    P = nAll * Cij ./(Ci' * Ci);
    S = log(P);
    S(S<0) = 0;
else
    fprintf('[Error] unkown similarity measure:%s\n', measure);
end
