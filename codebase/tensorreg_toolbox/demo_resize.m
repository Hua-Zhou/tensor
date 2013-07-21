%% resizing an array using different methods

clear;
% load an RBG image: 135-by-198-by-3 array
I = imread('onion.png');
disp(size(I));
figure; imshow(I);

% set target dimension
targetdim = round([100 150 3]);
disp(targetdim);

% downsize using interpolation
I_interp = array_resize(I, targetdim);
figure; imshow(I_interp);

% downsize using discrete cosine transform (DCT)
I_dct = array_resize(I, targetdim, 'method', 'dct');
figure; imshow(I_dct);

% downsize to PCA scores using HOSVD
I_hosvd = array_resize(I, targetdim, 'method', 'hosvd');
figure; imshow(I_hosvd);

% downsize to PCA scores marginal SVD
I_2dsvd = array_resize(I, targetdim, 'method', '2dsvd');
figure; imshow(I_2dsvd);

%% a downsizing-analysis-upsizing example

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);

% 2D true signal 64-by-64: cross
shape = imread('cross.gif'); 
shape = imresize(shape,[32,32]); % 32-by-32
b = zeros(2*size(shape));
b((size(b,1)/4):(size(b,1)/4)+size(shape,1)-1, ...
    (size(b,2)/4):(size(b,2)/4)+size(shape,2)-1) = shape;
[p1,p2] = size(b);
% true coefficients for regular (non-array) covariates
p0 = 5;
b0 = ones(p0,1);

% simulate covariates
n = 500;    % sample size
X = randn(n,p0);   % n-by-p0 regular design matrix
M = tensor(randn(p1,p2,n));  % p1-by-p2-by-n matrix variates
% the systematic part
mu = X*b0 + double(ttt(tensor(b), M, 1:2));
% simulate responses
sigma = 1;  % noise level
y = mu + sigma*randn(n,1);

% resize 2D covariates to 32-by-32
M_reduce = array_resize(double(M), [32 32 n], 'method', 'dct');

% rank 2 Kruskal regression with 64-by-64 covariates
tic;
[~,beta1,glmstats1] = kruskal_reg(X,M,y,2,'normal');
toc;

% rank 2 Kruskal regression with 32-by-32 covariates
tic;
[~,beta2,glmstats2] = kruskal_reg(X,M_reduce,y,2,'normal');
toc;
disp(size(beta2));
% resize estimate to original size
beta2 = array_resize(double(beta2), [64 64], 'method', 'dct');
disp(size(beta2));

% display true and recovered signals
figure; hold on;
set(gca,'FontSize',20);

subplot(1,3,1);
imagesc(-b);
colormap(gray);
title('True Signal');
axis equal;
axis tight;

subplot(1,3,2);
imagesc(-double(beta1));
colormap(gray);
title('Estimate using 64x64 covariates');
axis equal;
axis tight;

subplot(1,3,3);
imagesc(-beta2);
colormap(gray);
title('Estimate using 32x32 covariates');
axis equal;
axis tight;
