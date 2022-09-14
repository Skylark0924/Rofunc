function demo_proMP_Fourier_sampling02
% Stochastic sampling with Fourier movement primitives (2D example from handwriting data)
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
nbData = 200; %Number of datapoints in a trajectory
nbSamples = 10; %Number of demonstrations
nbVar = 2; %Dimension of position data (here: x1,x2)


% %% Generate periodic data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for n=1:nbSamples
% 	%Simulate variations on phase
% 	t = linspace(.6*n/3, 4*pi+.6*n/3, nbData);
% 	x(:,n) = kron((1+n.*1E-3) .* cos(t) + (.4+n.*1E-3) * cos(t*2+pi/3) + randn(1,nbData) .* 1E-4, ones(1,nbVar));
% end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/B.mat');
for n=1:nbSamples
	dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData/2)); %Resampling
	dTmp2 = fliplr(dTmp);
	x(:,n) = [dTmp(:); dTmp2(:)]; 
end

%Simulate phase variations
for n=2:nbSamples
	x(:,n) = circshift(x(:,1), (n-nbSamples/2-1)*4);
end


%% ProMP with Fourier basis functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = -8:8;
t = linspace(0,1,nbData);
phi = exp(t' * k * 2 * pi * 1i) ./ nbData;
Psi = kron(phi, eye(nbVar)); 
w = pinv(Psi) * x;
nbFct = size(w,1);

%Distribution in parameter space
Mu_m = mean(abs(w), 2); %Magnitude average
Mu_p = mean_angle(angle(w), 2); %Phase average
Mu_w = Mu_m .* exp(1i * Mu_p); %Reconstruction
Mu = Psi * Mu_w; %Reconstruction

Sigma_m = cov(abs(w')); %Magnitude spread
Sigma_p = cov_angle(angle(w')); %Phase spread
% Sigma_w = Sigma_m .* expm(1i * Sigma_p); % + eye(size(Sigma_m)) * 1E-4; 

[V_m, D_m] = eig(Sigma_m);
U_m = V_m * D_m.^.5;

[V_p, D_p] = eig(Sigma_p);
U_p = V_p * D_p.^.5;


%% Stochastic reproductions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbRepros = 20;

%Reproductions with learned variations
xw = (repmat(Mu_m,1,nbRepros) + U_m * randn(nbFct,nbRepros)) .* exp(1i * (Mu_p + U_p * randn(nbFct,nbRepros)));
xr = Psi * xw;

% %Reproductions with varied phases
% rTmp = linspace(0,1,nbRepros);
% xw = Mu_w .* exp(1i .* k' * rTmp);
% xr = Psi * xw;

% %Reproductions with varied magnitudes
% rTmp = linspace(1,2,nbRepros);
% xw = Mu_w .* rTmp;
% xr = Psi * xw;


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot signal
figure('position',[10 10 1500 800],'color',[1,1,1]); 
for i=1:nbVar
	axLim = [1, nbData, min(real(Mu(i:nbVar:end)))-1, max(real(Mu(i:nbVar:end)))+1];
	subplot(2,3,(i-1)*3+[1:2]); hold on;
	for n=1:nbSamples
		h(1) = plot(x(i:nbVar:end,n), '-','lineWidth',6,'color',[.7 .7 .7]);
	end
	for n=1:nbRepros
		h(2) = plot(real(xr(i:nbVar:end,n)), '-','lineWidth',2,'color',[1 .7 .7]);
		plot(imag(xr(i:nbVar:end,n)), ':','lineWidth',2,'color',[1 .7 .7]);
	end
	h(3) = plot(real(Mu(i:nbVar:end)), '-','lineWidth',4,'color',[.8 0 0]);
	h(4) = plot(imag(Mu(i:nbVar:end)), ':','lineWidth',4,'color',[.8 0 0]);
	patch([nbData/2 nbData/2 nbData+2 nbData+2], [axLim([3,4]), axLim([4,3])], [1 1 1],'edgecolor','none','facecolor',[1 1 1],'facealpha',.5);
	plot([nbData/2 nbData/2], axLim(3:4), ':','lineWidth',2,'color',[0 0 0]);

	axis(axLim);
	set(gca,'xtick',[],'ytick',[],'linewidth',2);
	xlabel('$t$','interpreter','latex','fontsize',25); ylabel('$x_1$','interpreter','latex','fontsize',25);
end
legend(h,{'Demonstrations','Stochastically generated samples','Re($\mu^x$)','Im($\mu^x$)'}, 'interpreter','latex','location','southeast','fontsize',22);

% %Plot magnitude and phase 
% clrmap = lines(nbFct);
% subplot(2,3,[3,6]); hold on; axis off;
% plot2DArrow([0;0], [.7;0].*nbData, [0,0,0], 1, .03.*nbData);
% plot2DArrow([0;0], [0;.7].*nbData, [0,0,0], 1, .03.*nbData);
% text((.7-.15).*nbData, .05.*nbData, 'Re($w$)','interpreter','latex','fontsize',20); text(.02.*nbData, (.7-.1).*nbData, 'Im($w$)','interpreter','latex','fontsize',20);
% nbDrawingSeg = 50;
% t = linspace(-pi, pi, nbDrawingSeg);
% xc = [cos(t); sin(t)];
% for i=1:nbFct
% 	R = diag([Sigma_p(i,i).^.5 + 1E-2, Sigma_m(i,i).^.5]);
% 	x = R * xc + repmat([Mu_p(i); Mu_m(i)], 1, nbDrawingSeg);
% 	[x(1,:), x(2,:)] = pol2cart(x(1,:), x(2,:));
% 	patch(x(1,:), x(2,:), clrmap(i,:),'edgecolor','none','facealpha',.3);		
% end
% for i=1:nbFct
% 	plot(real(w(i,:)), imag(w(i,:)), '.','markersize',25,'color',clrmap(i,:));
% 	for j=1:size(w,2)
% 		patch([0 real(w(i,j)) 0], [0 imag(w(i,j)) 0], clrmap(i,:), 'edgecolor',clrmap(i,:),'facealpha',.3,'edgealpha',.3);
% 	end
% 	plot(real(Mu_w(i)), imag(Mu_w(i)), '.','markersize',25,'color',clrmap(i,:));
% 	plot(real(Mu_w(i)), imag(Mu_w(i)), 'o','markersize',12,'linewidth',2,'color',clrmap(i,:));
% 	patch([0 real(Mu_w(i)) 0], [0 imag(Mu_w(i)) 0], clrmap(i,:), 'linewidth',4,'edgecolor',clrmap(i,:),'facealpha',.3,'edgealpha',.3);
% end
% axis equal; 

%2D plot
subplot(2,3,[3,6]); hold on; axis off;
for n=1:nbSamples
	plot(x(1:nbVar:nbData,n), x(2:nbVar:nbData,n), '-','lineWidth',8,'color',[.7 .7 .7]);
end
plot(real(Mu(1:nbVar:nbData)), real(Mu(2:nbVar:nbData)), '-','lineWidth',6,'color',[.8 0 0]);
for n=1:nbRepros
	plot(real(xr(1:nbVar:nbData,n)), real(xr(2:nbVar:nbData,n)), '-','lineWidth',2,'color',[1 .7 .7]);
end
axis equal;

% print('-dpng','graphs/demo_proMP_Fourier_sampling01.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%
function Mu = mean_angle(phi, dim)
	if nargin<2
		dim = 1;
	end
	Mu = angle(mean(exp(1i*phi), dim));
end

%%%%%%%%%%%%%%%%%%
function Sigma = cov_angle(phi)
	Mu = mean_angle(phi);
	e = phi - repmat(Mu, size(phi,1), 1);
	Sigma = cov(angle(exp(1i*e)));
end
