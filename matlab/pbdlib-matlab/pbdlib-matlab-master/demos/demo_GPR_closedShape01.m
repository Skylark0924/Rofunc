function demo_GPR_closedShape01
% Closed shape modeling with sequential points and periodic RBF kernel in Gaussian process regression (GPR) 
% 
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
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
nbVarX = 1; %Dimension of x
nbVarY = 2; %Dimension of y 
% nbVar = nbVarX + nbVarY; %Dimension of datapoint (x,y)
% nbData = 16; %Number of datapoints
nbDataRepro = 100; %Number of datapoints in a reproduction
nbRepros = 50; %Number of randomly sampled reproductions
p(1)=1E-1; p(2)=1E-1; p(3)=1E-5; %GPR parameters (here, for squared exponential kernels and noisy observations)
% SigmaY = [5, 2; 2, .1];
SigmaY = eye(nbVarY);


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y = randn(nbVarY, nbData) * 1E-1;
y = [-3 -2 -1  0  1  2  3  3  3  2  1  1  1  0 -1 -1 -1 -2 -3 -3;
		  2  2  2  2  2  2  2  1  0  0  0 -1 -2 -2 -2 -1  0  0  0  1] .*1E-2;
nbData = size(y,2);
x = linspace(0+.5/nbData, 1-.5/nbData, nbData);

Data = [x; y];


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPR precomputation (here, with a naive implementation of the inverse)
K = covFct(x, x, p, 1); %Inclusion of noise on the inputs

%Mean trajectory computation
xs = linspace(0,1,nbDataRepro);
Ks = covFct(xs, x, p);
r(1).Data = [xs; (Ks / K * y')']; 

%Uncertainty evaluation
Kss = covFct(xs, xs, p); 
S = Kss - Ks / K * Ks';
r(1).SigmaOut = zeros(nbVarY,nbVarY,nbData);
for t=1:nbDataRepro
% 	r(1).SigmaOut(:,:,t) = eye(nbVarY) * S(t,t); 
	r(1).SigmaOut(:,:,t) = SigmaY * S(t,t); 
end

%Generate stochastic samples from the prior 
[V,D] = eig(Kss);
for n=2:nbRepros/2
	yp = real(V*D^.5) * randn(nbDataRepro,nbVarY) * SigmaY .* 2E-1; 
	r(n).Data = [xs; yp'];
end

%Generate stochastic samples from the posterior 
[V,D] = eig(S);
for n=nbRepros/2+1:nbRepros
	ys = real(V*D^.5) * randn(nbDataRepro,nbVarY) * SigmaY .* 0.5 + r(1).Data(nbVarX+1:end,:)'; 
	r(n).Data = [xs; ys'];
end


%% Timeline plots 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 14 8],'position',[10 10 2500 1300]); 
limAxes = [0, 1, -.04 .04];

%Prior samples
subplot(2,3,1); hold on; title('Samples from prior','fontsize',12);
for n=2:nbRepros/2
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
% 	plot(r(n).Data(1,:), randn(1,nbDataRepro).*5E-2, '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
end
set(gca,'xtick',[],'ytick',[]); axis([0,1,-.2,.2]);
xlabel('x_1'); ylabel('y_1');

%Posterior samples
subplot(2,3,2); hold on;  title('Samples from posterior','fontsize',12);
for n=nbRepros/2+1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
end
plot(Data(1,:), Data(2,:), '.','markersize',18,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Trajectory distribution
subplot(2,3,3); hold on;  title('Trajectory distribution','fontsize',12);
patch([r(1).Data(1,:), r(1).Data(1,end:-1:1)], ...
	[r(1).Data(2,:)+squeeze(r(1).SigmaOut(1,1,:).^.5)', r(1).Data(2,end:-1:1)-squeeze(r(1).SigmaOut(1,1,end:-1:1).^.5)'], ...
	[.8 .8 .8],'edgecolor','none');
plot(r(1).Data(1,:), r(1).Data(2,:), '-','lineWidth',1,'color',[0 0 0]);
plot(Data(1,:), Data(2,:), '.','markersize',18,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');


%% Spatial plots 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Prior samples
subplot(2,3,4); hold on; axis off; 
for n=2:nbRepros/2
	plot(r(n).Data(2,:), r(n).Data(3,:), '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
end
axis equal;

%Posterior samples
subplot(2,3,5); hold on; axis off; 
for n=nbRepros/2+1:nbRepros
	plot(r(n).Data(2,:), r(n).Data(3,:), '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
end
plot(Data(2,:), Data(3,:), '.','markersize',18,'color',[1 0 0]);
axis equal;

%Trajectory distribution
subplot(2,3,6); hold on; axis off; 
plotGMM(r(1).Data(2:3,:), r(1).SigmaOut, [.8 .8 .8], .1);
plot(r(1).Data(2,:), r(1).Data(3,:), '-','lineWidth',1,'color',[0 0 0]);
plot(Data(2,:), Data(3,:), '.','markersize',18,'color',[1 0 0]);
axis equal;
% print('-dpng','graphs/GPR_closedShape05.png');
% print('-dpng','graphs/GPR_tmp01.png');

% figure; hold on; axis off; 
% colormap(flipud(gray));
% imagesc(abs(Kss));
% axis tight; axis square; axis ij;
% print('-dpng','graphs/GPR_kernel_closedShape01.png');

% figure; hold on; axis off; 
% for n=2:nbRepros/2
% 	plot(r(n).Data(2,:), r(n).Data(3,:), '-','lineWidth',1,'color',[.9 .9 .9]*rand(1));
% end
% axis equal; axis tight;
% print('-dpng','graphs/GPR_closedShape01.png');


pause;
close all;
end

% %User-defined distance function
% function d = distfun(x1, x2)
% % 	d = (x1*x2').^4 + (x1*x2').^3 + (x1*x2').^2 + x1*x2' + 1E-1;
% 	d = exp(-1E1 .* (x1-x2).^2);
% end

function K = covFct(x1, x2, p, flag_noiseObs)
	if nargin<4
		flag_noiseObs = 0;
	end

% 	%Kernel with user-defined distance function
% 	K = pdist2(x1', x2', @distfun); 

	%Standard periodic RBF kernel functions
	K = p(1) .* exp(-p(2)^-1 .* sin(pi.*pdist2(x1',x2')).^2);
	
% 	%Fixed periodic RBF kernel functions (see e.g. Eqs (15.88)+(16.1) p.534 of Kevin Murphy's book)
% 	nbStates = 50;
% 	Mu = linspace(0,1,nbStates);
% 	h1 = p(1) .* exp(-p(2)^-1 .* sin(pi.*pdist2(Mu',x1')).^2);
% 	h2 = p(1) .* exp(-p(2)^-1 .* sin(pi.*pdist2(Mu',x2')).^2);
% % 	h1 = h1 ./ repmat(sum(h1,1),nbStates,1);
% % 	h2 = h2 ./ repmat(sum(h2,1),nbStates,1);
% 	K = h1' * h2; %The rank of K is at most nbStates
	
	if flag_noiseObs==1
		K = K + p(3) * eye(size(x1,2),size(x2,2)); %Consideration of noisy observation y
	end
end