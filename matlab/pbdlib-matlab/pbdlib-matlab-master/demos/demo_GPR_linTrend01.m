function demo_GPR_linTrend01
% Gaussian process regression (GPR) with linear trend
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
nbVar = 2; %Dimension of datapoint (t,x1)
nbData = 4; %Number of datapoints
nbDataRepro = 100; %Number of datapoints for reproduction
nbRepros = 20; %Number of reproductions
p(1)=1E0; p(2)=1E-1; p(3)=1E-3; %GPR parameters


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 5E-1;
xIn = linspace(0,1,nbData);
Mu = A * xIn;
Data = [xIn; Mu + randn(1,nbData) * 1E-2];
xOut = Data(2:end,:);

%GPR precomputation
K = p(1) * exp(-p(2)^-1 * pdist2(xIn', xIn').^2);
invK = pinv(K + p(3) * eye(size(K))); 


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Mean trajectory computation
xInHat = linspace(0,1,nbDataRepro);
MuHat = A * xInHat;
Kd = p(1) * exp(-p(2)^-1 .* pdist2(xInHat', xIn').^2);
r(1).Data = [xInHat; MuHat + (Kd * invK * (xOut-Mu)')']; 
%Covariance computation
Kdd = p(1) * exp(-p(2)^-1 * pdist2(xInHat',xInHat').^2);
%Kdd = Kdd + p(3) * eye(size(Kdd)); 
S = Kdd - Kd * invK * Kd';
r(1).SigmaOut = zeros(nbVar-1,nbVar-1,nbData);
for t=1:nbDataRepro
	r(1).SigmaOut(:,:,t) = eye(nbVar-1) * S(t,t); 
end

%Generate stochastic samples from the prior 
[V,D] = eig(Kdd);
for n=2:nbRepros/2
	DataOut = MuHat' + real(V*D^.5) * randn(nbDataRepro,1)*2E-1;  
	r(n).Data = [xInHat; DataOut'];
end

%Generate stochastic samples from the posterior 
[V,D] = eig(S);
for n=nbRepros/2+1:nbRepros
	DataOut = real(V*D^.5) * randn(nbDataRepro,1)*0.5 + r(1).Data(2,:)';  
	r(n).Data = [xInHat; DataOut'];
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 12 4],'position',[10 10 1300 600]); 
%Prior samples
subplot(1,3,1); hold on; title('Samples from prior','fontsize',14);
for n=2:nbRepros/2
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end
set(gca,'xtick',[],'ytick',[]); axis([0, 1, -.5 1]);
%xlabel('$x_1$','interpreter','latex','fontsize',18);
%ylabel('$y_1$','interpreter','latex','fontsize',18);
xlabel('$x^{\scriptscriptstyle\mathcal{I}}_1$','interpreter','latex','fontsize',18);
ylabel('$x^{\scriptscriptstyle\mathcal{O}}_1$','interpreter','latex','fontsize',18);

%Posterior samples
subplot(1,3,2); hold on;  title('Samples from posterior','fontsize',14);
for n=nbRepros/2+1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis([0, 1, -.5 1]);
%xlabel('$x_1$','interpreter','latex','fontsize',18);
%ylabel('$y_1$','interpreter','latex','fontsize',18);
xlabel('$x^{\scriptscriptstyle\mathcal{I}}_1$','interpreter','latex','fontsize',18);
ylabel('$x^{\scriptscriptstyle\mathcal{O}}_1$','interpreter','latex','fontsize',18);

%Trajectory distribution
subplot(1,3,3); hold on;  title('Trajectory distribution','fontsize',14);
patch([r(1).Data(1,:), r(1).Data(1,end:-1:1)], ...
	[r(1).Data(2,:)+squeeze(r(1).SigmaOut.^.5)', r(1).Data(2,end:-1:1)-squeeze(r(1).SigmaOut(:,:,end:-1:1).^.5)'], ...
	[.8 .8 .8],'edgecolor','none');
plot(r(1).Data(1,:), r(1).Data(2,:), '-','lineWidth',3.5,'color',[0 0 0]);
plot(Data(1,:), Data(2,:), '.','markersize',24,'color',[1 0 0]);
set(gca,'xtick',[],'ytick',[]); axis([0, 1, -.5 1]);
%xlabel('$x_1$','interpreter','latex','fontsize',18);
%ylabel('$y_1$','interpreter','latex','fontsize',18);
xlabel('$x^{\scriptscriptstyle\mathcal{I}}_1$','interpreter','latex','fontsize',18);
ylabel('$x^{\scriptscriptstyle\mathcal{O}}_1$','interpreter','latex','fontsize',18);

%print('-dpng','-r300','graphs/GPR_RBF_Ax_1E0_1E-1_1E-3.png');
pause;
close all;

