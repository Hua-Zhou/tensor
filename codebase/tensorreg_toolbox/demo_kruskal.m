%% 2D covariates?linear regression

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);

% 2D true signal 64-by-64: b
shape = imread('cross.gif'); 
shape = imresize(shape,[32,32]); % 32-by-32
b = zeros(2*size(shape));
b((size(b,1)/4):(size(b,1)/4)+size(shape,1)-1, ...
    (size(b,2)/4):(size(b,2)/4)+size(shape,2)-1) = shape;
[p1,p2] = size(b);
% true coefficients 
p0 = 5;
b0 = ones(p0,1);

% simulate covariates
n = 500;    % sample size
X = randn(n,p0);   % n-by-p regular design matrix
M = randn(p1,p2,n);  % p1-by-p2-by-n matrix variates
% the systematic part
mu = X*b0 + squeeze(sum(sum(repmat(b,[1 1 n]).*M,1),2));
% simulate responses
sigma = 1;  % noise level
y = normrnd(mu,sigma,n,1);

% estimate without using array covariates
[beta_rk0,dev0] = glmfit(X,y,'normal','constant','off');

% estimate using tensor regression - rank 1
tic;
disp('rank 1');
[beta0_rk1,beta_rk1,glmstats1,dev1] = kruskal_reg(X,M,y,1,'normal');
toc;

% estimate using tensor regression - rank 2
tic;
disp('rank 2');
[beta0_rk2,beta_rk2,glmstats2,dev2] = kruskal_reg(X,M,y,2,'normal');
toc;

% estimate using tensor regression - rank 3
tic;
disp('rank 3');
[beta0_rk3,beta_rk3,glmstats3,dev3] = kruskal_reg(X,M,y,3,'normal');
toc;

% display true and recovered signals
figure; hold on;
set(gca,'FontSize',20);

subplot(1,4,1);
imagesc(-b);
colormap(gray);
title('True Signal');
axis equal;
axis tight;

subplot(1,4,2);
imagesc(-double(beta_rk1));
colormap(gray);
title({['TR(1),', ' BIC=',num2str(glmstats1{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,3);
imagesc(-double(beta_rk2));
colormap(gray);
title({['TR(2),', ' BIC=',num2str(glmstats2{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,4);
imagesc(-double(beta_rk3));
colormap(gray);
title({['TR(3),', ' BIC=',num2str(glmstats3{end}.BIC)]});
axis equal;
axis tight;

%% 2D covariates?sparse linear regression

%% 2D covariates?logistic regression

%% 2D covariates?sparse logistic regression

%% 3D covaraites, linear regression

%% 3D covaraites, sparse linear regression

%% 3D covariates, logistic regression

%% 3D covariates, sparse logistic regression
