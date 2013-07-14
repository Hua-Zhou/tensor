function [beta0,B,stats] = matrix_sparsereg(X,M,y,lambda,dist,varargin)
% MATRIX_SPARSEREG Fit nuclear norm regularized matrix regression
%
% INPUT:
%   X - n-by-p0 regular covariate matrix
%   M - matrix variates (or tensors) with dim(M) = [p1,p2,n]
%   y - n-by-1 respsonse vector
%   lambda - regularization parameter
%   dist - 'binomial', 'gamma', 'inverse gaussian','normal', or 'poisson'
%
% Output:
%   beta0 - regression coefficients for the regular covariates
%   B - regression coefficients for matrix variates
%   stats - algorithmic statistics

% COPYRIGHT: North Carolina State University
% AUTHOR: Hua Zhou (hua_zhou@ncsu.edu), Lexin Li

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('y', @isnumeric);
argin.addRequired('lambda', @isnumeric);
argin.addRequired('dist', @(x) ischar(x));
argin.addParamValue('B0', [], @(x) isa(x,'ktensor') || isempty(x));
argin.addParamValue('delta', 1e-3, @(x) isnumeric(x) && x>0);
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('penalty', 'enet', @ischar);
argin.addParamValue('penparam', 1, @isnumeric);
argin.addParamValue('TolFun', 1e-3, @(x) isnumeric(x) && x>0);
argin.addParamValue('Warnings', 'off', @ischar);
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,M,y,lambda,dist,varargin{:});

B0 = argin.Results.B0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
pentype = argin.Results.penalty;
penparam = argin.Results.penparam;
ridgedelta = argin.Results.delta;
TolFun = argin.Results.TolFun;
Warnings = argin.Results.Warnings;
wts = argin.Results.weights;

% check model and response variable
if (strcmpi(dist,'normal'))
    model = 'LINEAR';
elseif (strcmpi(dist,'binomial'))
    model = 'LOGISTIC';
    if (any(y<0) || any(y>1))
        error('responses outside [0,1]');
    end
elseif (strcmpi(dist,'poisson'))
    model = 'LOGLINEAR';
    if (any(y<0))
        error('responses y must be nonnegative');
    end
else
    error('model not recogonized. LINEAR|LOGISTIC|LOGLINEAR accepted');
end

% check dimensions
if (length(size(M))~=3)
    error('M should be of size p1-by-p2-by-n');
end
[p1,p2,n] = size(M);
if (isempty(X))
    X = ones(n,1);  % intercept term
end
if (size(X,1)~=n)
    error('X should be n-by-p0');
end

% default weights
if (isempty(wts))
    wts = ones(n,1);
end

% convert M into a tensor T (if it's not)
TM = tensor(M);
TMn = tenmat(TM,3,[1,2]);

% initialize
alpha_old = 0; alpha = 1;
if (isempty(B0))
    B = ktensor(0,{zeros(p1,1),zeros(p2,1)});
else
    B = B0;
end
TB = full(B);
B_old = B;
objval = inf;


% turn off warnings
if (strcmpi(Warnings,'off'))
warning('off','stats:glmfit:IterationLimit');
warning('off','stats:glmfit:BadScaling');
warning('off','stats:glmfit:IllConditioned');
end

% main loop
isdescent = true;
for iter=1:MaxIter

    % current search point
    if isdescent
        S = TB+(alpha_old-1)/alpha*(TB-full(B_old));
    else
        S = full(B_old)+(alpha_old/alpha)*(TB-full(B_old));
    end
    innerS = double(TMn*tenmat(S,[1 2]));
    [beta0,dev0,glmstats] = glmfit_priv(X,y,dist,'constant','off', ...
        'weights',wts,'offset',innerS);
    innerX = X*beta0;
    lossS = dev0/2;
    lossD1S = ttv(TM,glmweights(innerX+innerS,y,wts,model),3);
    
    % line search
    B_old = B;
    objval_old = objval;
    for l=1:50
        A = double(S)-ridgedelta*double(lossD1S);
        [U,s,V] = svt(A,ridgedelta*lambda,...
            'pentype',pentype,'penparam',penparam);
        if (isempty(s))
            stats.maxlambda = ...
                lsq_maxlambda(1,-svds(A,1),pentype,penparam)/ridgedelta;
            stats.AIC = dev0 + 2*size(X,2);
            stats.BIC = dev0 + log(n)*size(X,2);
            stats.yhat = y - glmstats.resid;
            return;
        end
        B = ktensor(s,{U,V});
        TB = full(B);
        innerB = double(TMn*tenmat(TB,[1 2]));
        % objective value
        [beta0,dev0,glmstats] = glmfit_priv(X,y,dist,'constant','off',...
            'weights',wts,'offset',innerB);
        objval = dev0/2 ...
            + sum(penalty_function(B.lambda,lambda,pentype,penparam));
        % surrogate value
        BminusS = TB-full(S);
        surval = lossS + innerprod(lossD1S,BminusS) ...
            + norm(BminusS)^2/2/ridgedelta ...
            + sum(penalty_function(B.lambda,lambda,pentype,penparam));
        % line search stopping rule
        if (objval<=surval)
            break;
        else
            ridgedelta = ridgedelta/2;
        end
    end
    
    % force descent
    if (objval<=objval_old) % descent
        % stopping rule
        if (abs(objval_old-objval)<TolFun*(abs(objval_old)+1))
            break;
        end
        isdescent = true;
    else % no descent
        objval = objval_old;
        if isdescent
            isdescent = false;
        else
            break;
        end        
    end
    
    % display
    if (~strcmpi(Display,'off'))
        display(['iter ' num2str(iter) ', objval=' num2str(objval)]);
    end

    % update alpha constants
    alpha_old = alpha;
    alpha = (1+sqrt(4+alpha_old^2))/2;
    
end
warning on all;

% collect algorithmic statistics
stats.yhat = y - glmstats.resid;
stats.iterations = iter;
Aspectrum = svd(A);
if (p1~=p2)
    Aspectrum(max(p1,p2)) = 0;
end
stats.dof = size(X,2);
for i=1:nnz(B.lambda)
    stats.dof = stats.dof + 1 ...
        + sum(Aspectrum(i)*(Aspectrum(i)-ridgedelta*lambda) ...
        ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:p1)].^2)) ...
        + sum(Aspectrum(i)*(Aspectrum(i)-ridgedelta*lambda) ...
        ./(Aspectrum(i)^2-[Aspectrum(1:i-1); Aspectrum(i+1:p2)].^2));
end
stats.AIC = dev0 + 2*stats.dof;
stats.BIC = dev0 + log(n)*stats.dof;

    function [glmwts] = glmweights(inner,y,wt,model)
        big = 20;
        switch upper(model)
            case 'LINEAR'
                glmwts = - wt.*(y-inner);
            case 'LOGISTIC'
                expinner = exp(inner);
                prob = expinner./(1+expinner);
                prob(inner>big) = 1;
                prob(inner<-big) = 0;
                glmwts = - wt.*(y-prob);
            case 'LOGLINEAR'
                expinner = exp(inner);
                glmwts = - wt.*(y-expinner);
        end
    end%GLMWEIGHTS

end