function demo_PCA01
% Principal component analysis (PCA)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarX = 3; %Dimension of the original data
nbVarU = 2; %Dimension of the subspace (number of principal components)
nbData = 50; %Number of datapoints


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r = rand(nbVarX,10) - .5;
[V,D] = eigs(cov(r'));
[~,id] = sort(diag(D),'descend');
D = D(id,id);
V = V(:,id);
D(nbVarU+1:end,nbVarU+1:end) = 0;
R = real(V*D.^.5);
b = (rand(nbVarX,1)-.5) * 2;
x = R * randn(nbVarX,nbData) + repmat(b,1,nbData) + randn(nbVarX,nbData) .* 1E-3;


%% PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[V,D] = eigs(cov(x'), nbVarU);
% [~,id] = sort(diag(D),'descend');
% D = D(id,id);
% V = V(:,id);
% A = V(:,1:nbVarU) * D(1:nbVarU,1:nbVarU).^.5; %Projection operator
A = V * D.^.5; %Projection operator
u = A \ x;


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1800,1200]); 
subplot(1,2,1); hold on; rotate3d on; title('Original space (3D)');
plot3(x(1,:), x(2,:), x(3,:), '.','markersize',8,'color',[.2 .2 .2]);
plotGMM3D(b, cov(x'), [.8 .8 .8], .5);
view(3); axis tight; axis equal; axis vis3d;
set(gca,'xtick',[],'ytick',[],'ztick',[]);
xlabel('x_1'); ylabel('x_2'); zlabel('x_2');

subplot(1,2,2); hold on; title('Subspace (2D)');
plot(u(1,:), u(2,:), '.','markersize',8,'color',[.2 .2 .2]);
axis equal; 
set(gca,'xtick',[],'ytick',[]);
xlabel('u_1'); ylabel('u_2');

%print('-dpng','graphs/demo_PCA01.png');
pause;
close all;