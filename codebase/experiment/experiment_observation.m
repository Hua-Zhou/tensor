

%% 2D Case

% clear all;
% 
% originB = get_img('test_data\cross.png');
% % True coefficients for regular (non-array) covariates
% p0 = 5;
% b0 = ones(p0, 1);
% 
% % Simulate covariates
% n = 500;    % sample size
% dimension = 2;
% X = randn(n, p0);   % n-by-p0 regular design matrix
% originM = tensor(randn([size(originB), n]));  % p1-by-p2-by-n matrix variates
% 
% % Simulate responses
% mu = X*b0 + double(ttt(tensor(originB), originM, 1:dimension));
% sigma = 1;  % noise level
% y = mu + sigma*randn(n,1);
% 
% l = 64:-8:16; %downsized size for every dimension
% vol = l.^dimension; %volume
% replicateSeq = fix(linspace(5, 13, length(l)));
% 
% rank = 2;
% 
% for i=1:length(l)
%     disp(['Size:', num2str(l(i)), '*', num2str(l(i))]);
%     
%     % interpolation
%     B = array_resize(originB, [l(i), l(i)]);
%     M = array_resize(originM, [l(i), l(i), n]);
%     
%     tic;
%     [~,beta,~,dev] = kruskal_reg_v2(X,M,y,rank,'normal','Replicates', replicateSeq(i));
%     elapseTime = toc;
%     
%     dist = abs(double(beta) - B);
%     dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
%     
%     subplot(4, length(l), i);
%     imagesc(-B);
%     colormap(gray);
%     title({['Size:', num2str(l(i)), '^2']});
%     axis equal;
%     axis tight;
%     axis off;
%     
%     subplot(4, length(l), length(l)+i);
%     imagesc(-double(beta));
%     colormap(gray);
%     title({['Avg. Dev.:', num2str(dev/vol(i), 4)];...
%         ['L1 Dist.:', num2str(dist/vol(i), 4)]});
%     axis equal;
%     axis tight;
%     axis off;
%     
%     % dct
%     B = array_resize(originB, [l(i), l(i)], 'method', 'dct');
%     M = array_resize(originM, [l(i), l(i), n], 'method', 'dct');
%     
%     tic;
%     [~,beta,~,dev] = kruskal_reg_v2(X,M,y,rank,'normal','Replicates', replicateSeq(i));
%     elapseTime = toc;
%     
%     dist = abs(double(beta) - B);
%     dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
%     
%     subplot(4, length(l), 2*length(l)+i);
%     imagesc(-B);
%     colormap(gray);
%     title({['Size:', num2str(l(i)), '^2']});
%     axis equal;
%     axis tight;
%     axis off;
%     
%     subplot(4, length(l), 3*length(l)+i);
%     imagesc(-double(beta));
%     colormap(gray);
%     title({['Avg. Dev.:', num2str(dev/vol(i), 4)];...
%         ['Avg. L1 Dist.:', num2str(dist/vol(i), 4)]});
% 
%     axis equal;
%     axis tight;
%     axis off;
% end
% 
% saveas(gcf, 'experiment\observation-2d.fig', 'fig');

%% 3D Case

% clear all;
% 
% originB = zeros(24, 24, 24);
% originB(4:8, 4:8, 4:8) = 1;
% originB(16:20, 16:20, 16:20) = 1;
% 
% % true coefficients for regular covariates
% p0 = 5;
% b0 = ones(p0, 1);
% 
% % Simulate covariates
% n = 500;    % sample size
% dimension = 3;
% X = randn(n, p0);   % n-by-p regular design matrix
% originM = tensor(randn([size(originB), n]));  % p1-by-p2-by-p3 3D variates
% 
% % Simulate responses
% mu = X*b0 + double(ttt(originM,tensor(originB),1:dimension));
% sigma = 1;  % noise level
% y = mu + sigma*randn(n,1);
% 
% l = 24:-4:8; % downsized size for every dimension
% vol = l.^dimension; % volume
% replicateSeq = fix(linspace(5, 13, length(l)));
% rank = 2;
% 
% for i=1:length(l)
%     disp(['Size:', num2str(l(i)), '*', num2str(l(i)), '*', num2str(l(i))]);
%     
%     % interpolation
%     B = array_resize(originB, [l(i), l(i), l(i)]);
%     M = array_resize(originM, [l(i), l(i), l(i), n]);
%     
%     tic;
%     [~,beta,~,dev] = kruskal_reg_v2(X,M,y,rank,'normal','Replicates', replicateSeq(i));
%     elapseTime = toc;
%     
%     dist = abs(double(beta) - B);
%     dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
%     
%     subplot(4, length(l), i);
%     view(3);
%     isosurface(B, 0.5);
%     xlim([1, l(i)]);
%     ylim([1, l(i)]);
%     zlim([1, l(i)]);
%     title({['Size:', num2str(l(i)), '^3']});
%     
%     subplot(4, length(l), length(l)+i);
%     view(3);
%     isosurface(double(beta), 0.5);
%     xlim([1, l(i)]);
%     ylim([1, l(i)]);
%     zlim([1, l(i)]);
%     title({['Avg. Dev.:', num2str(dev / vol(i), 4)];...
%         ['Avg. L1 Dist.:', num2str(dist / vol(i), 4)]});
%     
%     %dct
%     B = array_resize(originB, [l(i), l(i), l(i)], 'method', 'dct');
%     M = array_resize(originM, [l(i), l(i), l(i), n], 'method', 'dct');
%     
%     tic;
%     [~,beta,~,dev] = kruskal_reg_v2(X,M,y,rank,'normal','Replicates', replicateSeq(i));
%     elapseTime = toc;
%     
%     dist = abs(double(beta) - B);
%     dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
%     
%     subplot(4, length(l), 2*length(l)+i);
%     view(3);
%     isosurface(B, 0.5);
%     xlim([1, l(i)]);
%     ylim([1, l(i)]);
%     zlim([1, l(i)]);
%     title({['Size:', num2str(l(i)), '^3']});
%     
%     subplot(4, length(l), 3*length(l)+i);
%     view(3);
%     isosurface(double(beta), 0.5);
%     xlim([1, l(i)]);
%     ylim([1, l(i)]);
%     zlim([1, l(i)]);
%     title({['Avg. Dev.:', num2str(dev / vol(i), 4)]; ['Avg. L1 Dist.:', num2str(dist / vol(i), 4)]});
% 
% end
% 
% saveas(gcf, 'experiment\observation-3d.fig', 'fig');

%% Simulation

clear all;

rank = 2;
dimension = 2;

% True coefficients for regular (non-array) covariates
p0 = 5;
b0 = ones(p0, 1);
p1 = 64;
p2 = 64;
n = 500;    % sample size
sigma = 1;  % noise level

l = 48:-16:16; %downsized size for every dimension
vol = l.^dimension; % volume
seqLen = length(l);
replicateSeq = fix(linspace(5, 13, seqLen));
simulationTime = 100;

dev = zeros(simulationTime, 2*seqLen); % estimation deviance
dist1 = zeros(simulationTime, 2*seqLen); % L1 distance between downsized groundtruth & estimation
dist2 = zeros(simulationTime, 2*seqLen); % L1 distance between groundtruth & upsized estimation

for t=1:simulationTime
    disp(['Time:', num2str(t)]);
    
    % Simulate covariates
    X = randn(n, p0);   % n-by-p0 regular design matrix
    originB = tensor(ktensor(ones(rank, 1), randn(p1, rank), randn(p2, rank)));    
    originM = tensor(randn([size(originB), n]));  % p1-by-p2-by-n matrix variates
    
    % Simulate responses
    mu = X*b0 + double(ttt(tensor(originB), originM, 1:dimension));
    y = mu + sigma*randn(n,1);    
    
    for i=1:seqLen
        disp(['Size:', num2str(l(i)), '*', num2str(l(i))]);
        
        %interpolation
        B = array_resize(originB, [l(i), l(i)]);
        M = array_resize(originM, [l(i), l(i), n]);
        
        [~,beta,~,dev(t,2*i-1)] = kruskal_reg_v2(X,M,y,rank,'normal',...
            'Replicates', replicateSeq(i));

        dist = abs(double(beta) - double(B));
        dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
        dist1(t,2*i-1) = dist;
        
        beta = array_resize(beta, size(originB));
        dist = abs(double(beta) - double(originB));
        dist = double(ttt(tensor(ones(size(originB))), tensor(dist), 1:dimension));
        dist2(t,2*i-1) = dist;
        
        % dct
        B = array_resize(originB, [l(i), l(i)], 'method', 'dct');
        M = array_resize(originM, [l(i), l(i), n], 'method', 'dct');
        
        [~,beta,~,dev(t,2*i)] = kruskal_reg_v2(X,M,y,rank,'normal',...
            'Replicates', replicateSeq(i));
        
        dist = abs(double(beta) - double(B));
        dist = double(ttt(tensor(ones(size(B))), tensor(dist), 1:dimension));
        dist1(t,2*i) = dist;
        
        beta = array_resize(beta, size(originB), 'method', 'dct');
        dist = abs(double(beta) - double(originB));
        dist = double(ttt(tensor(ones(size(originB))), tensor(dist), 1:dimension));
        dist2(t,2*i) = dist;
    end
end

tmp = zeros(1, 2*seqLen);
tmp(logical(mod(1:2*seqLen, 2))) = vol;
tmp(logical(~mod(1:2*seqLen, 2))) = vol;

dev = bsxfun(@rdivide, dev, tmp);
dist1 = bsxfun(@rdivide, dist1, tmp);
dist2 = bsxfun(@rdivide, dist2, tmp);

boxlabel = cell(1, 2*seqLen);
for i=1:seqLen
    boxlabel{2*i-1} = ['i. ', num2str(l(i))];
    boxlabel{2*i} = ['d. ', num2str(l(i))];
end

subplot(1, 3, 1);
boxplot(dev, boxlabel);
title('Avg. estimation dev.', 'FontSize', 8);

subplot(1, 3, 2);
boxplot(dist1, boxlabel);
title('          Avg. dist. between downsized\newline          groundtruth & estimation', 'FontSize', 8);

subplot(1, 3, 3);
boxplot(dist2, boxlabel);
title('                  Avg. dist. between groundtruth\newline                  & upsized estimation',...
    'FontSize', 8);

saveas(gcf, 'experiment\simulation.fig', 'fig');
save('experiment\simulation.mat', 'dev', 'dist1', 'dist2', 'rank', 'n', 'p0',...
    'b0', 'p1', 'p2', 'dimension', 'sigma', 'simulationTime', 'l', 'replicateSeq');