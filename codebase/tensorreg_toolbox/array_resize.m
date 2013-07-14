function [B] = array_resize(A, targetdim, varargin)
% ARRAY_RESIZE Resize array A to [[A;K1,K2,...,KD]] where K is specified
%   by method
%
% INPUT
%   A: a D-dimensional array
%   targetdim: a 1-by-D vector of target dimensions
%   method: name of kernel method
%
% OUTPUT
%   B: resized array
%
% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu)

% parse inputs
argin = inputParser;
argin.addRequired('A');
argin.addRequired('targetdim', @isnumeric);
argin.addParamValue('method', [], @(x) ischar(x)||isempty(x));
argin.addParamValue('inverse', false, @islogical);
argin.parse(A, targetdim, varargin{:});
isinverse = argin.Results.inverse;
method = argin.Results.method;

% for 2D input, the default is image resize by Matlab imresize() function
if (isempty(method))
    B = imresize(A,targetdim);
    return;
end

% turn array into tensor structure
A = tensor(A);
D = ndims(A);
p = size(A);

% argument checking
if (length(targetdim)~=D)
    error('targetdim does not match dimension of A');
end

% obtain the support of the wavelets
[hr] = wfilters(method,'r');
N = fix(length(hr)/2);

% U holds the component matrices in Tucker tensor
U = cell(1,D);
for d=1:D
    if (isinverse)
        Kd = wpfun(method,p(d)-1,ceil(log2(targetdim(d)/(2*N-1))));
        Kd = bsxfun(@times, Kd, 1./sqrt(sum(Kd.^2,2)));
        U{d} = imresize(Kd,[p(d),targetdim(d)])';
    else
        Kd = wpfun(method,targetdim(d)-1,ceil(log2(p(d)/(2*N-1))));
        Kd = bsxfun(@times, Kd, 1./sqrt(sum(Kd.^2,2)));
        U{d} = imresize(Kd,[targetdim(d),p(d)]);
    end
end
B = double(ttensor(A,U));

end