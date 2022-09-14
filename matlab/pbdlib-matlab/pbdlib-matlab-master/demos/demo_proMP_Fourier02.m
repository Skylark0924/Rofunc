function demo_proMP_Fourier02
% ProMP with Fourier basis functions (2D example)
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
% 	author="Calinon, S.",
% 	title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
% 	booktitle="Mixture Models and Applications",
% 	publisher="Springer",
% 	editor="Bouguila, N. and Fan, W.", 
% 	year="2019",
% 	pages="39--57",
% 	doi="10.1007/978-3-030-23876-6_3"
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
nbVar = 2; %Dimension of position data (here: x1,x2)
nbData = 200; %Number of datapoints in a trajectory
nbSamples = 4; %Number of demonstrations


%% Generate periodic data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbSamples
	t = linspace(1*n/3, 4*pi+1*n/3, nbData);
	xtmp = (1+n.*1E-1) .* [cos(t); sin(t)] + (.7+n.*1E-1) .* [zeros(1,nbData); cos(t*2-pi/3)] + randn(nbVar,nbData) .* 1E-2;
	x(:,n) = xtmp(:);
end


%% ProMP with Fourier basis functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute basis functions Psi and activation weights w
k = -5:5;
nbFct = length(k);
t = linspace(0,1,nbData);
phi = exp(t' * k * 2 * pi * 1i) / nbData;
Psi = kron(phi, eye(nbVar)); 

% w = (Psi' * Psi + eye(nbFct).*1E-18) \ Psi' * x; 
w = pinv(Psi) * x; 

%Distribution in parameter space
Mu_R = mean(abs(w), 2); %Magnitude average
Mu_theta = mean_angle(angle(w), 2); %Phase average
Mu_w = Mu_R .* exp(1i * Mu_theta); %Reconstruction

Sigma_R = cov(abs(w')); %Magnitude spread
Sigma_theta = cov_angle(angle(w')); %Phase spread
% Sigma_w = Sigma_R .* exp(1i * Sigma_theta)  + eye(size(Sigma_R)) * 1E0; %Reconstruction

%Trajectory distribution
Mu = Psi * Mu_w; 
% Sigma = Psi * Sigma_w * Psi'; 


%% Stochastic reproductions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbRepros = 20;
[V_R, D_R] = eig(Sigma_R);
[V_theta, D_theta] = eig(Sigma_theta);
U_R = V_R * D_R.^.5;
U_theta = V_theta * D_theta.^.5;
xr_R = repmat(Mu_R, 1, nbRepros) + U_R * randn(nbFct*nbVar,nbRepros) .* 9E-1; %Magnitude sampling
xr_theta = repmat(Mu_theta, 1, nbRepros) + U_theta * randn(nbFct*nbVar,nbRepros) .* 9E-1; %Phase sampling
xr_w = xr_R .* exp(1i * xr_theta); %Reconstruction
xr = Psi * xr_w; %real(2 * Psi(:,2:end) * xr_w(2:end,:)) + Psi(:,1) * xr_w(1,:);


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1200 800]); 
clrmap = lines(nbFct);

%Plot signal 2D
subplot(4,2,[1:2:7]); hold on; axis off; 
for n=1:nbSamples 
	plot(x(1:2:end,n), x(2:2:end,n), '-','lineWidth',4,'color',[.7 .7 .7]);
end
for n=1:nbRepros
	plot(real(xr(1:2:end,n)), real(xr(2:2:end,n)), '-','lineWidth',1,'color',[1 .7 .7]);
	plot(imag(xr(1:2:end,n)), imag(xr(2:2:end,n)), ':','lineWidth',1,'color',[1 .7 .7]);
end
plot(real(Mu(1:2:end)), real(Mu(2:2:end)), '-','lineWidth',3,'color',[.8 0 0]);
axis tight; axis equal; 

%Plot signal 1D
subplot(4,2,2); hold on; %axis off; %x1 part
id = 1:2:nbData*nbVar;
for n=1:nbSamples 
	plot(x(id,n), '-','lineWidth',4,'color',[.7 .7 .7]);
end
for n=1:nbRepros 
	plot(real(xr(id,n)), '-','lineWidth',1,'color',[1 .7 .7]);
	plot(imag(xr(id,n)), ':','lineWidth',1,'color',[1 .7 .7]);
end
% std = real(diag(Sigma(id,id))'.^.5) .* 3;
% msh = [[1:nbData, nbData:-1:1, 1]; [real(Mu(id)')+std, fliplr(real(Mu(id)')-std), real(Mu(id(1)))+std(1)]];
% patch(msh(1,:), msh(2,:), [.8 0 0],'edgecolor','none','facealpha',.3); 
h(1) = plot(real(Mu(id)), '-','lineWidth',3,'color',[.8 0 0]);
h(2) = plot(imag(Mu(id)), ':','lineWidth',3,'color',[.8 0 0]);
axis([1, nbData, min(x(1:2:end,1))-.8, max(x(1:2:end,1))+.8]); %axis tight; 
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',22); ylabel('$x_1$','interpreter','latex','fontsize',22);
legend(h,{'Re($\mu^x$)','Im($\mu^x$)'},'interpreter','latex','location','southeast','fontsize',16);

subplot(4,2,4); hold on; %axis off; %x2 part
id = 2:2:nbData*nbVar;
for n=1:nbSamples 
	plot(x(id,n), '-','lineWidth',4,'color',[.7 .7 .7]);
end
for n=1:nbRepros 
	plot(real(xr(id,n)), '-','lineWidth',1,'color',[1 .7 .7]);
	plot(imag(xr(id,n)), ':','lineWidth',1,'color',[1 .7 .7]);
end
% std = real(diag(Sigma(id,id))'.^.5) .* 3;
% msh = [[1:nbData, nbData:-1:1, 1]; [real(Mu(id)')+std, fliplr(real(Mu(id)')-std), real(Mu(id(1)))+std(1)]];
% patch(msh(1,:), msh(2,:), [.8 0 0],'edgecolor','none','facealpha',.3); 
plot(real(Mu(id)), '-','lineWidth',3,'color',[.8 0 0]);
plot(imag(Mu(id)), ':','lineWidth',3,'color',[.8 0 0]);
axis([1, nbData, min(x(2:2:end,1))-.8, max(x(2:2:end,1))+.8]); %axis tight; 
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',22); ylabel('$x_2$','interpreter','latex','fontsize',22);

%Plot basis functions
subplot(4,2,6); hold on; %axis off; %real part
for i=1:nbFct
	plot(1:nbData, real(phi(:,i)),'-','linewidth',2,'color',clrmap(i,:));
end
axis([1, nbData, min(real(phi(:))), max(real(phi(:)))]); %axis tight; 
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',22); ylabel('Re($\phi_k$)','interpreter','latex','fontsize',22);

subplot(4,2,8); hold on; %axis off; %imaginary part
for i=1:nbFct
	plot(1:nbData, imag(phi(:,i)),':','linewidth',2,'color',clrmap(i,:));
end
axis([1, nbData, min(imag(phi(:)))-1E-4, max(imag(phi(:)))+1E-4]); %axis tight; 
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',22); ylabel('Im($\phi_k$)','interpreter','latex','fontsize',22);

% print('-dpng','graphs/proMP_Fourier02.png');
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