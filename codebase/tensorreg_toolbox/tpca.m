function [Mpca,PC,latent] = tpca(M,d)
% TPCA Tensor principle component analysis
%
% [MPCA, PC, LATENT] = TPCA(M,d) performs the tensor version of the
%   traditional PCA. Given n tensor observations, flatten each along mode i
%   to obtain a matrix, take outer product of the matrix (p_i-by-p_i), then
%   do the classical PCA on the sum of outer products, retrieve the first
%   d_i principal components (p_i-by-d_i), then scale the original tensor
%   in the PC basis along each mode.
%
%   INPUT:
%       M: array variates (aka tensors) with dim(M) = [p_1,p_2,...,p_D,n]
%       d: target dimensions [d_1,...,d_D]
%
%   Output:
%       MPCA: array with dim(Mpca)=[d_1,...,d_D] after change of basis. In
%           PCA literature, it is called the SCOREs
%       PC: D-by-1 cell array containing principal components along each
%           mode. PC{i} has dimension p_i-by-d_i. In PCA literature, they
%           are called the COEFFICIENTs
%       LATENT: D-by-1 cell array containing the d_i eigen values along
%           each mode
%
% Examples
%
% See also pca
%
% TODO
%   - need to provide the option of centering
%
% COPYRIGHT 2011-2013 North Carolina State University
% Hua Zhou <hua_zhou@ncsu.edu>

% check dimensionalities
p = size(M); n = p(end); p(end) = []; D = length(p);
if (size(d,1)>size(d,2))
    d = d';
end
if length(d)~=D
    error('tensorreg:tpca:wrongdim', ...
        'target dimensions do not match array dimension');
end
if any(d>p)
    error('tensorreg:tpca:exceeddim', ...
        'target dimensions cannot exceed original dimensions');
end

% change M to tensor (if it is not)
TM = tensor(M);

% loop over dimensions to obtain PCs
PC = cell(1,D);
latent = cell(1,D);
idx = repmat({':'}, D, 1);
for dd=1:D
    C = zeros(p(dd),p(dd)); % p_d-by-p_d
    for i=1:n
        tmati = double(tenmat(TM(idx{:},i),dd));   % #rows = p_d
        C = C + tmati*tmati';
    end
    C = C/n;
    [PC{dd},latent{dd}] = eigs(C,d(dd));
    latent{dd} = diag(latent{dd});
end

% change of basis for original array data
Mpca = ttm(tensor(M),[cellfun(@(X) X',PC,'UniformOutput',false),eye(n)]);
Mpca = double(Mpca);

end
