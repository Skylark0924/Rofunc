function demo_kernelPCA01
% Kernel PCA, with comparison to PCA.
%
% If this code is useful for your research, please cite the related publication:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarY = 2; %Dimension of subspace
nbData = 2000; %Number of datapoints


%% Generate 3D swiss-roll data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = 3.*pi./2 .* (1 + 2.*rand(1,nbData));  
x = [t.*cos(t); 60.*rand(1,nbData); t.*sin(t)] + 1E-1.*randn(3,nbData);
x = x - repmat(mean(x,2),1,nbData); %Recenter data


%% Kernel PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = covFct(x,x); 

%Recenter K
M = ones(size(K)) ./ size(x,2);
K = K - M*K - K*M + M*K*M;

%Eigendecomposition
[V,D] = eigs(K, nbVarY);
for i=1:nbVarY
	V(:,i) = V(:,i) ./ D(i,i)^.5;
end
[~,id] = sort(diag(D),'descend'); 
V = V(:,id);

%Projection of the data in the lower dimensional space
y = V' * K'; 


%% PCA (for comparison)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[V,D] = eigs(cov(x'), nbVarY); %Eigendecomposition
% [~,id] = sort(diag(D),'descend'); 
% V = V(:,id);
% D = D(id,id);
A = V * D.^.5;
y_pca = A \ x;


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 2300 1000]); 
%Original data
subplot(1,3,1); axis off; hold on; rotate3d on; title('Original data');
plot3(x(1,:), x(2,:), x(3,:), 'k.');
axis vis3d; view(-5,15);
%PCA
subplot(1,3,2); axis off; hold on; title('PCA');
plot(y_pca(1,:), y_pca(2,:), 'b.');
axis equal; 
%Kernel PCA
subplot(1,3,3); axis off; hold on; title('Kernel PCA');
plot(y(1,:), y(2,:), 'r.');
axis equal;

pause;
close all;
end

% %User-defined distance function
% function d = distfun(x,y)
% 	d = (x*y' + 1E-2).^2;
% end

function K = covFct(x1, x2)
% 	K = pdist2(x1', x2', @distfun); %User-defined distance function
	K = exp(-1E-2 .* pdist2(x1',x2').^2); %Squared exponential kernels
end