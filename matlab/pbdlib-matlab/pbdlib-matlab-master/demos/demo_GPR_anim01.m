function demo_GPR_anim01
% Gaussian process regression (GPR) with prior distribution formed so that
% it can be smoothly animated for illustration purpose
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
nbVar = nbVarX+1; %Dimension of datapoint (x,y)
nbData = 4; %Number of datapoints
nbDataRepro = 100; %Number of datapoints in a reproduction
nbRepros = 50; %Number of randomly sampled reproductions
p(1)=1E0; p(2)=1E-1; p(3)=1E-3; %GPR parameters (here, for squared exponential kernels and noisy observations)


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = linspace(0,1,nbData);
y = randn(1,nbData) * 1E-1;
Data = [x; y];


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPR precomputation (here, with a naive implementation of the inverse)
K = covFct(x, x, p, 1); %Inclusion of noise on the inputs

%Mean trajectory computation
xs = linspace(0,1,nbDataRepro);
Ks = covFct(xs, x, p);
r0.Data = [xs; (Ks / K * y')']; 

%Uncertainty evaluation
Kss = covFct(xs, xs, p); 
S = Kss - Ks / K * Ks';
r0.SigmaOut = zeros(nbVar-1,nbVar-1,nbData);
for t=1:nbDataRepro
	r0.SigmaOut(:,:,t) = eye(nbVarX) * S(t,t); 
end

%Generate stochastic samples from the prior 
[V,D] = eig(Kss);
nbp = 3;
u = spline(1:nbp, randn(nbDataRepro,nbp)*2E-1, linspace(1,nbp,nbRepros));
for n=1:nbRepros
	yp = real(V*D^.5) * u(:,n);
	r(n).Data = [xs; yp'];
end

%Generate stochastic samples from the posterior 
[V,D] = eig(S);
u = spline(1:nbp, randn(nbDataRepro,nbp)*5E-1, linspace(1,nbp,nbRepros));
for n=1:nbRepros
	ys = real(V*D^.5) * u(:,n) + r0.Data(2,:)'; 
	r2(n).Data = [xs; ys'];
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 22 4],'position',[10 10 2300 600]); 
limAxes = [0, 1, -.5 .5];

%Prior samples
subplot(1,4,1); hold on; title('Smooth samples from prior','fontsize',14);
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',1,'color',[.8 .8 .8]*n/nbRepros); 
end
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Posterior samples
subplot(1,4,2); hold on; title('Samples from posterior','fontsize',14);
for n=1:nbRepros
	plot(r2(n).Data(1,:), r2(n).Data(2,:), '-','lineWidth',1,'color',[.8 .8 .8]*n/nbRepros);
end
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Trajectory distribution
subplot(1,4,3); hold on; title('Trajectory distribution','fontsize',14);
patch([r0.Data(1,:), r0.Data(1,end:-1:1)], ...
	[r0.Data(2,:)+squeeze(r0.SigmaOut.^.5)', r0.Data(2,end:-1:1)-squeeze(r0.SigmaOut(:,:,end:-1:1).^.5)'], ...
	[.8 .8 .8],'edgecolor','none');
plot(r0.Data(1,:), r0.Data(2,:), '-','lineWidth',1,'color',[0 0 0]);
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis(limAxes);
xlabel('x_1'); ylabel('y_1');

%Plot covariance
subplot(1,4,4); hold on; axis off; title('Covariance','fontsize',14);
colormap(flipud(gray));
imagesc(abs(Kss));
axis tight; axis square; axis ij;

pause;
close all;
end


% %User-defined distance function
% function d = distfun(x1,x2)
% % 	d = min(x1,x2) + .1;
% % 	d = (x1*x2' + 1E-1).^2;
% % 	d = exp(-1E1 .* (x1-x2).^2);
% end


function K = covFct(x1, x2, p, flag_noiseObs)
	if nargin<4
		flag_noiseObs = 0;
	end
	
	%RBF covariance function
	K = p(1) .* exp(-p(2)^-1 .* pdist2(x1',x2').^2);

% 	%User-defined covariance function
% 	K = pdist2(x1', x2', @distfun); %User-defined distance function

% 	%Covariance function with RBFs at fixed positions
% 	nbStates = 3;
% 	Mu = linspace(0,1,nbStates);
% 	h1 = exp(-3E1 .* pdist2(Mu',x1').^2); 
% 	h2 = exp(-3E1 .* pdist2(Mu',x2').^2);
% % 	h1 = h1 ./ repmat(sum(h1,1),nbStates,1);
% % 	h2 = h2 ./ repmat(sum(h2,1),nbStates,1);
% 	K = h1' * h2;

% 	figure; hold on;
% 	for i=1:nbStates
% 		plot(h1(i,:));
% 	end
% 	pause

	if flag_noiseObs==1
		K = K + p(3) * eye(size(x1,2),size(x2,2)); %Consideration of noisy observation y
	end
end