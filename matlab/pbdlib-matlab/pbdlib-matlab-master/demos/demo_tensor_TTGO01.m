function demo_tensor_TTGO01
% Global optimization with tensor trains (TTGO), consisting of:
% - Encoding the cost as a distribution with a low-rank factorization using TT-cross algorithm
% - (Conditional) sampling from the tensor train decomposition of the distribution 
% (see https://sites.google.com/view/ttgo for details about the approach and for more elaborated codes in Python)
%
% If this code is useful for your research, please cite the related publication:
% @article{Shetty22,
%   author={Shetty, S. and Lembono, T. and L\"ow, T. and Calinon, S.},
%   title={Tensor Trains for Global Optimization Problems in Robotics},
%   journal={arXiv:2206.05077},
%   year={2022}
% }
%
% Copyright (c) 2022 Idiap Research Institute, https://idiap.ch/
% Written by Sylvain Calinon, https://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


%% Parameters and data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r = 3; %Rank for tensor decomposition
rTT = [1, r, r]; %TT-rank vector
n_max = 500; %Maximum number of iterations

nbStates = 3; %Number of Gaussians
Mu = [1 2 2; 3 6 2; 4 1 3]; %Means 
sigma = [1, 1, 1] * 1E0; %Variances

%Generating distribution as a GMM
nbVar = [5,6,4];
x = zeros(nbVar);
for i=1:nbVar(1)
	for j=1:nbVar(2)
		for k=1:nbVar(3)
			for l=1:nbStates
				eTmp = [i; j; k] - Mu(:,l);
				x(i,j,k) = x(i,j,k) + exp(-eTmp'*eTmp/sigma(l)) / nbStates;
			end
		end
	end
end
x = x + rand(nbVar) * 1E-5;


%% TT-cross approximation of the distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iv{1} = randi(prod(nbVar(2:end)), [r,1]); %Initial column list
iv{2} = randi(prod(nbVar(3:end)), [r,1]); %Initial column list

Pmat{1} = reshape(x, [nbVar(1), prod(nbVar(2:end))]); 
for n=1:n_max
	%Forward pass (select rows)
	for k=1:length(nbVar)-1
		U = qr(Pmat{k}(:,iv{k}));
		iu{k} = maxvol(U(:,1:r)); %Select rows
		Pmat{k+1} = reshape(Pmat{k}(iu{k},:), [nbVar(k+1)*rTT(k+1), prod(nbVar(k+2:end))]); 
	end
	%Backward pass (select columns)
	for k=length(nbVar)-1:-1:1
		V = qr(Pmat{k}(iu{k},:)');
		iv{k} = maxvol(V(:,1:r)); %Select columns
	end
end
%disp(['TT-cross converged in ' num2str(n) ' iterations.']);

%Reconstruct TT-cores
for k=1:length(nbVar)-1
	P{k} = reshape(Pmat{k}(:,iv{k}) / Pmat{k}(iu{k},iv{k}), [rTT(k) nbVar(k) rTT(k+1)]);
end
P{length(nbVar)} = reshape(Pmat{end}(:,1), [rTT(end) nbVar(end)]);

%Reconstruct the tensor from the TT-cores
x_est = zeros(nbVar);
for i=1:nbVar(1)
	for j=1:nbVar(2)
		for k=1:nbVar(3)
			x_est(i,j,k) = reshape(P{1}(1,i,:), [1,size(P{1},3)]) * reshape(P{2}(:,j,:), [size(P{2},1),size(P{2},3)]) * reshape(P{3}(:,k,1), [size(P{3},1),1]);
		end
	end
end


%% TT sampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = .2; %Prioritized sampling factor
N = 500; %Number of sampled points
d = length(nbVar); %Order of tensor

Pi_hat{d} = 1;
for k=d-1:-1:1
	Pi_hat{k} = squeeze(sum(P{k+1},2)) * Pi_hat{k+1};
end
phi{1} = ones(N,1);
for k=1:d
	Pi=[];
	for xk=1:size(P{k},2)
		Pi(:,xk) = reshape(P{k}(:,xk,:), [size(P{k},1),size(P{k},3)]) * Pi_hat{k};
	end
	for l=1:N
		p = abs(phi{k}(l,:) * Pi)';		
		p = p / max(p);
		p = p.^(1/(1-alpha+1E-9)); %Prioritized sampling
		p = p / sum(p);
		id(k,l) = find(rand<cumsum(p), 1, 'first'); %Sampling from multinomial distribution (indices)
		phi{k+1}(l,:) = phi{k}(l,:) * reshape(P{k}(:,id(k,l),:), [size(P{k},1),size(P{k},3)]);
	end
end

%Sampled data (from indices)
x_gen = zeros(nbVar);
for l=1:N
	x_gen(id(1,l), id(2,l), id(3,l)) = x_gen(id(1,l), id(2,l), id(3,l)) + 1; 
end
x_gen = x_gen * max(x(:)) / max(x_gen(:)); %max(x(:)) is given instead of 1 for color plots on the same scale


%% Plots with flattened 3D tensors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('color',[1 1 1],'position',[10,10,1200,1400]); hold on; axis off;
colormap(flipud(gray));

%Original tensor
mplot(x(:,:,1), [-7, 0]);
mplot(x(:,:,2), [-7, 7]);
mplot(x(:,:,3), [-7, 14]);
mplot(x(:,:,4), [-7, 21]);
text(-7,-3, 'Original tensor','horizontalalignment','center','fontsize',30);

%Estimated tensor
mplot(x_est(:,:,1), [0, 0]);
mplot(x_est(:,:,2), [0, 7]);
mplot(x_est(:,:,3), [0, 14]);
mplot(x_est(:,:,4), [0, 21]);
text(0,-3, 'Estimated tensor','horizontalalignment','center','fontsize',30);

%Tensor data samples
mplot(x_gen(:,:,1), [7, 0]);
mplot(x_gen(:,:,2), [7, 7]);
mplot(x_gen(:,:,3), [7, 14]);
mplot(x_gen(:,:,4), [7, 21]);
text(7,-3, ['Samples (N=' num2str(N) ')'],'horizontalalignment','center','fontsize',30);

axis ij; axis equal; axis tight; 
%print('-dpng','graphs/TTsampling01.png'); %'-r300',

waitfor(h);
end

%% Maximal volume submatrix in an tall matrix based on LU decomposition
%% (returns rows indices that contain maximal volume submatrix)
function id = maxvol(A)
	nbMaxIter = 100;
	[n, r] = size(A); 
	%Initialization
	[~,~,p] = lu(A,'vector');
	id = p(1:r);
	B = A / A(id,:);
	%Iterative algorithm
	for i=1:nbMaxIter
		[mx0, id2] = max(abs(B(:)));
		[i0, j0] = ind2sub([n,r], id2);
		if (mx0 <= 1 + 5e-2) 
			id = sort(id);
			break;	
		end 
		k = id(j0); %This is the current row that we are using
		B = B + B(:,j0) * (B(k,:) - B(i0,:)) / B(i0,j0);
		id(j0) = i0;
	end
end

%% Matrix plot function
function h = mplot(c, pos, alpha)
	if nargin < 3
		alpha = 1;
	end	
	if nargin < 2
		pos = [0,0];
	end
	pos = pos - [size(c,2), size(c,1)] * 0.5 + 0.5;
	imagesc([0, size(c,2)-1] + pos(1), [0, size(c,1)-1] + pos(2), c); 
	rgx = [0:size(c,2)] - 0.5 + pos(1);
	rgy = [0:size(c,1)] - 0.5 + pos(2);
	line(repmat(rgx, 2, 1), repmat([rgy(1); rgy(end)], 1, length(rgx)), 'color', [0 0 0]); % vertical
	line(repmat([rgx(1); rgx(end)], 1, length(rgy)), repmat(rgy, 2, 1), 'color', [0 0 0]); % horizontal
end
