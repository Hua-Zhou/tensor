%% 2D covariates, linear regression

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);

% 2D true signal 64-by-64
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
y = mu + sigma*randn(n,1);

% estimate using tensor regression - rank 1
tic;
disp('rank 1');
[~,beta_rk1,glmstats1] = kruskal_reg(X,M,y,1,'normal');
toc;

% estimate using tensor regression - rank 2
tic;
disp('rank 2');
[~,beta_rk2,glmstats2] = kruskal_reg(X,M,y,2,'normal');
toc;

% estimate using tensor regression - rank 3
tic;
disp('rank 3');
[~,beta_rk3,glmstats3] = kruskal_reg(X,M,y,3,'normal');
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
title({['rank=1, ', ' BIC=',num2str(glmstats1{end}.BIC,5)]});
axis equal;
axis tight;

subplot(1,4,3);
imagesc(-double(beta_rk2));
colormap(gray);
title({['rank=2 ', ' BIC=',num2str(glmstats2{end}.BIC,5)]});
axis equal;
axis tight;

subplot(1,4,4);
imagesc(-double(beta_rk3));
colormap(gray);
title({['rank=3, ', ' BIC=',num2str(glmstats3{end}.BIC,5)]});
axis equal;
axis tight;

%% 2D covariates, sparse linear regression

% set lasso penalty and tuning parameter values
pentype = 'enet';
penparam = 1;
lambda = [0,100,1000];

% estimate using tensor regression - lambda 1
tic;
disp(['lambda=', num2str(lambda(1))]);
[~,beta_rk1,~,glmstat_rk1] = kruskal_sparsereg(X,M,y,3,'normal',lambda(1),...
    pentype,penparam);
toc;

% estimate using tensor regression - lambda 2
tic;
disp(['lambda=', num2str(lambda(2))]);
[~,beta_rk2,~,glmstat_rk2] = kruskal_sparsereg(X,M,y,3,'normal',lambda(2),...
    pentype,penparam);
toc;

% estimate using tensor regression - lambda 3
tic;
disp(['lambda=', num2str(lambda(3))]);
[~,beta_rk3,~,glmstat_rk3] = kruskal_sparsereg(X,M,y,3,'normal',lambda(3),...
    pentype,penparam);
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
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(1))];...
    ['BIC=', num2str(glmstat_rk1{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,3);
imagesc(-double(beta_rk2));
colormap(gray);
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(2))];...
    ['BIC=', num2str(glmstat_rk2{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,4);
imagesc(-double(beta_rk3));
colormap(gray);
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(3))];...
    ['BIC=', num2str(glmstat_rk3{end}.BIC)]});
axis equal;
axis tight;

%% 2D covariates, logistic regression

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);

% 2D true signal 64-by-64
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
n = 1000;    % sample size
X = randn(n,p0);   % n-by-p regular design matrix
M = randn(p1,p2,n);  % p1-by-p2-by-n matrix variates
% the systematic part
mu = X*b0 + squeeze(sum(sum(repmat(b,[1 1 n]).*M,1),2));
% simulate binoary responses from the systematic components
y = binornd(1, 1./(1+exp(-mu)));

% estimate using tensor regression - rank 1
tic;
disp('rank 1');
[beta0_rk1,beta_rk1,glmstats1,dev1] = kruskal_reg(X,M,y,1,'binomial');
toc;

% estimate using tensor regression - rank 2
tic;
disp('rank 2');
[beta0_rk2,beta_rk2,glmstats2,dev2] = kruskal_reg(X,M,y,2,'binomial');
toc;

% estimate using tensor regression - rank 3
tic;
disp('rank 3');
[beta0_rk3,beta_rk3,glmstats3,dev3] = kruskal_reg(X,M,y,3,'binomial');
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
title({['rank=1, ', ' BIC=',num2str(glmstats1{end}.BIC,5)]});
axis equal;
axis tight;

subplot(1,4,3);
imagesc(-double(beta_rk2));
colormap(gray);
title({['rank=2 ', ' BIC=',num2str(glmstats2{end}.BIC,5)]});
axis equal;
axis tight;

subplot(1,4,4);
imagesc(-double(beta_rk3));
colormap(gray);
title({['rank=3, ', ' BIC=',num2str(glmstats3{end}.BIC,5)]});
axis equal;
axis tight;

%% 2D covariates, sparse logistic regression

% set lasso penalty and tuning parameter values
pentype = 'enet';
penparam = 1;
lambda = [0,10,50];

% estimate using tensor regression - lambda 1
tic;
disp(['lambda=', num2str(lambda(1))]);
[~,beta_rk1,~,glmstat_rk1] = kruskal_sparsereg(X,M,y,3,'binomial',lambda(1),...
    pentype,penparam);
toc;

% estimate using tensor regression - lambda 2
tic;
disp(['lambda=', num2str(lambda(2))]);
[~,beta_rk2,~,glmstat_rk2] = kruskal_sparsereg(X,M,y,3,'binomial',lambda(2),...
    pentype,penparam);
toc;

% estimate using tensor regression - lambda 3
tic;
disp(['lambda=', num2str(lambda(3))]);
[~,beta_rk3,~,glmstat_rk3] = kruskal_sparsereg(X,M,y,3,'binomial',lambda(3),...
    pentype,penparam);
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
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(1))];...
    ['BIC=', num2str(glmstat_rk1{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,3);
imagesc(-double(beta_rk2));
colormap(gray);
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(2))];...
    ['BIC=', num2str(glmstat_rk2{end}.BIC)]});
axis equal;
axis tight;

subplot(1,4,4);
imagesc(-double(beta_rk3));
colormap(gray);
title({['TR(3),' pentype '(' num2str(penparam), '), \lambda=', num2str(lambda(3))];...
    ['BIC=', num2str(glmstat_rk3{end}.BIC)]});
axis equal;
axis tight;

%% 3D covaraites, linear regression

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);

% true 3D signal: 'two-patch'
b = zeros(25,25,25);
b(6:10,6:10,6:10) = 1;
b(18:22,18:22,18:22) = 1;
[p1, p2, p3] = size(b);

figure; 
isosurface(b,.5); 
xlim([1, p1]);
ylim([1, p2]);
zlim([1, p3])
title('True Signal');

% true regression coefficients for regular covariates
p0 = 5;
b0 = ones(p0,1);

% simulate covariates
n = 500;    % sample size
X = randn(n,p0);   % n-by-p regular design matrix
M = tensor(randn(p1,p2,p3,n));  % p1-by-p2-by-p3 3D variates
% the systematic part
mu = X*b0 + double(ttt(M,tensor(b),1:3));
% simulate responses
sigma = 1;  % noise level
y = mu + sigma*randn(n,1);

%% estimate by Kruskal regression - rank 1

tic;
disp('rank 1');
[~,beta_rk1,glmstats1] = kruskal_reg(X,M,y,1,'normal');
toc;

figure;
isosurface(double(beta_rk1),0.5); 
title({['rank=1, ', ' BIC=',num2str(glmstats1{end}.BIC,5)]});

% estimate by Kruskal regression - rank 2
tic;
disp('rank 2');
[~,beta_rk2,glmstats2] = kruskal_reg(X,M,y,2,'normal');
toc;

figure;
isosurface(double(beta_rk2),0.5); 
title({['rank=2, ', ' BIC=',num2str(glmstats2{end}.BIC,5)]});

% estimate by Kruskal regression - rank 3
tic;
disp('rank 3');
[~,beta_rk3,glmstats3] = kruskal_reg(X,M,y,3,'normal');
toc;

figure;
isosurface(double(beta_rk3),0);
title({['rank=3, ', ' BIC=',num2str(glmstats3{end}.BIC,5)]});

%% 3D covaraites, sparse linear regression

