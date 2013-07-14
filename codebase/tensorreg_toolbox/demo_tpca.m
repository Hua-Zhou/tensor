%% generate a random array (better to use a real example?)

clear;
% reset random seed
s = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(s);
% simulate array data
p = [6 8 10];   % array size
n = 100;        % # observations
d = [2 2 2];    % # PCs requested
ra = rand([p n]);
% tensor PCA
[score, coeff, latent] = tpca(ra, d);
disp(size(score));
disp(coeff{1});
disp(latent{1});
