function [result, sampRpp] = correlatedGaussianNoise(Rpp, nSamp)
%% generates correlated 0-mean Gaussian vector process
%
% inputs: Rpp - correlation matrix. must be positive definite
%               and size determines output vector
%         nSamp - number of independent samples of correlated process
%
% output: result - matrix of dimension Rpp rows X nSamp cols.
%         sampRpp - sample correlation matrix
%
% result has form:
%        < ------ nSamp ------->
%        ------------------------   data is correlated along
%   |    |                      |   all rows for each column
%   |    |       output         |
%   p    |    data  matrix      |   data is independent along
%   |    |                      |   all columns for a given row
%   |    |                      |
%        < -------- nSamp ------>
%
% example: use following correlation matrix (output implicitly 3 rows) 
%          [1    0.2   0.2] 
%          [0.2   1    0.2] 
%          [0.2  0.2    1 ]
% 
% and 1e5 samples to check function
%
%% 
% n = 3; Rpp = repmat(0.2, [n,n]);  Rpp(1:(n+1):n^2) = 1;
% disp(Rpp)
% nSamp = 1e5;
% [x, sampR] = correlatedGaussianNoise(Rpp, nSamp);
% disp(sampR)
%
%% -----------------------------------------------------
% michaelB
%

%% algorithm

% check dimenions - add other checking as necessary...
if(ndims(Rpp) ~= 2),
    result = [];
    error('Rpp must be a real-valued symmetric matrix');
    return;
end

% symmeterize the correlation matrix
Rpp = 0.5 .*(double(Rpp) + double(Rpp'));

% eigen decomposition
[V,D] = eig(Rpp);

% check for positive definiteness
if(any(diag(D) <= 0)),
    result = [];
    error('Rpp must be a positive definite');
    return;
end

% form correlating filter
W = V*sqrt(D);

% form white noise dataset
n = randn(size(Rpp,1), nSamp);

% correlated (colored) noise
result = W * n;

% calculate the sample correlation matrix if necessary
if(nargout == 2),
    sampRpp = 0;
    for k1=1:nSamp,
        sampRpp = sampRpp + result(:,k1) * (result(:,k1)');
    end
    
    % unbiased estimate
    sampRpp = sampRpp ./ (nSamp - 1);
end
