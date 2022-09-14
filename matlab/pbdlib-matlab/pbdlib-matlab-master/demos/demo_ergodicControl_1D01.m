function demo_ergodicControl_1D01
% 1D ergodic control with a spatial distribution described as a GMM, inspired by G. Mathew and I. Mezic, 
% "Spectral Multiscale Coverage: A Uniform Coverage Algorithm for Mobile Sensor Networks", CDC'2009 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 5000; %Number of datapoints
nbFct = 10; %Number of basis functions along x and y
nbStates = 8; %Number of Gaussians to represent the spatial distribution
dt = 1E-2; %Time step
xlim = [0; 1]; %Domain limit for each dimension (considered to be 1 for each dimension in this implementation)
L = (xlim(2) - xlim(1)) * 2; %Size of [-xlim(2),xlim(2)]
om = 2 * pi / L; %omega
u_max = 1E0; %Maximum speed allowed 

%Desired spatial distribution represented as a mixture of Gaussians
% Mu(:,1) = 0.7;
% Sigma(:,1) = 0.003; 
% Mu(:,2) = 0.5;
% Sigma(:,2) = 0.01;
Mu = rand(1, nbStates);
Sigma = repmat(1E-1, [1,1,nbStates]);
Priors = ones(1,nbStates) ./ nbStates; %Mixing coefficients


%% Compute Fourier series coefficients w_hat of desired spatial distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rg = [0:nbFct-1]';
kk = rg .* om;
Lambda = (rg.^2 + 1).^-1; %Weighting vector (Eq.(15)

% tic
%Explicit description of w_hat by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries)
w_hat = zeros(nbFct,1);
for j=1:nbStates
	w_hat = w_hat + Priors(j) .* cos(kk .* Mu(:,j)) .* exp(-.5 * kk.^2 .* Sigma(:,j)); %Eq.(22)
end
w_hat = w_hat / L;
% toc
% return

% %Alternative computation of w_hat by discretization (for verification)
% nbRes = 100;
% x = linspace(xlim(1), xlim(2), nbRes); %Spatial range
% g = zeros(1,nbRes);
% for k=1:nbStates
% 	g = g + Priors(k) .* mvnpdf(x', Mu(:,k)', Sigma(:,k))'; %Spatial distribution 
% end
% phi_inv = cos(kk * x) ./ L ./ nbRes;
% w_hat = phi_inv * g'; %Fourier coefficients of spatial distribution
% 
% % w_hat = zeros(nbFct,1);
% % for n=1:nbFct
% % 	w_hat(n) = sum(g .* cos(x .* kk(n))); 
% % end
% % w_hat = w_hat ./ L ./ nbRes; %Fourier coefficients of spatial distribution


%Fourier basis functions (for a discretized map)
nbRes = 200;
xm = linspace(xlim(1), xlim(2), nbRes); %Spatial range
phim = cos(kk * xm) * 2; %Fourier basis functions
phim(2:end,:) = phim(2:end,:) * 2;

%Desired spatial distribution 
g = w_hat' * phim;

% %Alternative computation of g
% g = zeros(1,nbRes);
% for k=1:nbStates
% 	g = g + Priors(k) .* mvnpdf(xm', Mu(:,k)', Sigma(:,k))'; %Spatial distribution
% 	%(2.*pi.*s1).^-.5 .*  exp(-.5 .* s1.^-1 .* (X - m1).^2);
% end


%% Ergodic control 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = .1; %Initial position
wt = zeros(nbFct, 1);
for t=1:nbData
	r.x(:,t) = x; %Log data
	
	%Fourier basis functions and derivatives for each dimension (only cosine part on [0,L/2] is computed since the signal is even and real by construction) 
	phi = cos(x * kk); %Eq.(18)
	dphi = -sin(x * kk) .* kk; %Gradient of basis functions

	wt = wt + phi / L;	%wt./t are the Fourier series coefficients along trajectory (Eq.(17))

% 	%Controller with ridge regression formulation
% 	u = -dphi' * (Lambda .* (wt./t - w_hat)) * t * 1E-1; %Velocity command
	
	%Controller with constrained velocity norm
	u = -dphi' * (Lambda .* (wt/t - w_hat)); %Eq.(24)
	u = u .* u_max / (norm(u)+1E-2); %Velocity command
	
	x = x + u * dt; %Update of position
	r.g(:,t) = (wt/t)' * phim; %Reconstructed spatial distribution (for visualization)
	r.w(:,t) = wt/t; %Fourier coefficients along trajectory (for visualization)
% 	r.e(t) = sum(sum((wt/t - w_hat).^2 .* Lambda)); %Reconstruction error evaluation
end


% vt = (vt+dt*phi)
% Bk = sum( hadamard_product(Lambda, (vt/t-w_hat)*t, dphi_k) )
% B = [Bk k=1,..,d]
% uk = -umax*Bk/norm(B), k=1,,,d



% %The Fourier series coefficients along trajectory can alternatively be computed in batch form
% % wt2 = sum(cos(r.x' * kk'))' ./ L;
% wt2 = cos(kk * r.x) * ones(nbData,1) ./ L;
% norm(wt - wt2)
% return


% %Alternative computation of desired and reconstructed spatial distribution 
% g = zeros(1,nbRes);
% for n=1:nbFct
% 	g = g + w_hat(n) .*  cos(xm .* kk(n));
% end
% r.g = zeros(1,nbRes);
% for n=1:nbFct
% 	r.g = r.g + (wt(n)./nbData) .*  cos(xm .* kk(n));
% end


%% Plot (static)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = nbData; %Reconstruction at time step s
T = nbData; %Size of time window
figure('position',[10,10,2200,1200],'color',[1 1 1]); 

%Plot distribution
subplot(3,2,1); hold on; 
h(1) = plot(xm, g, '-','linewidth',4,'color',[.8 0 0]);
h(2) = plot(xm, r.g(:,s), '-','linewidth',2,'color',[0 0 0]);
% plot(xm, (r.g(:,s)-r.g(:,s-1)).*1E2, '-','linewidth',3,'color',[0 0 .8]); %Plot increment (scaled)
legend(h,{'Desired','Reconstructed'},'fontsize',18,'location','northwest');
axis([xlim', -.3, max(g)*1.2]);
set(gca,'xtick',[],'ytick',[],'linewidth',2);
xlabel('$x$','interpreter','latex','fontsize',28); 
ylabel('$g(x)$','interpreter','latex','fontsize',28);  

%Plot Fourier coefficients
subplot(3,2,2); hold on; 
plot(rg, zeros(nbFct,1), '-','linewidth',1,'color',[0 0 0]);
plot(rg, w_hat, '.','markersize',26,'color',[.8 0 0]);
for n=1:nbFct
	plot([rg(n), rg(n)], [0, w_hat(n)], '-','linewidth',4,'color',[.8 0 0]);
end
plot(rg, r.w(:,s), '.','markersize',18,'color',[0 0 0]);
for n=1:nbFct
	plot([rg(n), rg(n)], [0, r.w(n,s)], '-','linewidth',2,'color',[0 0 0]);
end
% plot(rg, (r.w(:,s)-r.w(:,s-1)).*1E2, 'o','linewidth',2,'markersize',7,'color',[0 0 .8]); %Plot increment (scaled)
axis([0, nbFct-1, min(w_hat)-5E-2, max(w_hat)+5E-2]);
set(gca,'xtick',[0,nbFct-1],'ytick',[],'linewidth',2,'fontsize',20);
xlabel('$k$','interpreter','latex','fontsize',28);
ylabel('$w_k$','interpreter','latex','fontsize',28); 

%Plot Lambda_k
subplot(3,2,4); hold on; 
plot(rg, Lambda, '-','linewidth',2,'color',[0 0 0]);
plot(rg, Lambda, '.','markersize',18,'color',[0 0 0]);
axis([0, nbFct-1, min(Lambda)-5E-2, max(Lambda)+5E-2]);
set(gca,'xtick',[0,nbFct-1],'ytick',[0,1],'linewidth',2,'fontsize',20);
xlabel('$k$','interpreter','latex','fontsize',28);
ylabel('$\Lambda_k$','interpreter','latex','fontsize',28); 

%Plot signal
subplot(3,2,[3,5]); hold on; box on;
plot(r.x(:,s-T+1:s), 1:T, '-','linewidth',3,'color',[0 0 0]);
plot(r.x(:,s), T, '.','markersize',28,'color',[0 0 0]);
axis([xlim', 1, T]);
set(gca,'linewidth',2,'xtick',[],'ytick',[1,T],'yticklabel',{'t-T','t'},'fontsize',24);
xlabel('$x$','interpreter','latex','fontsize',28);   
% print('-dpng','graphs/ergodicControl_1D01.png'); 


% %% Additional plots (static)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %Plot error
% % figure; hold on;
% % plot(r.e(200:end));
% 
% % %Plot decomposition of desired distribution
% % dg = 4;
% % figure('position',[10,10,900,1200],'color',[1 1 1]); hold on; axis off;
% % %Plot distribution
% % plot(xm, g*2, '-','linewidth',3,'color',[.8 0 0]);
% % for n=1:6
% % 	plot(xm, .3*phim(n,:)-dg*n,'-','linewidth',2,'color',[0 .6 0]);
% % end
% % % print('-dpng','graphs/ergodicControl_1DbasisFcts01.png'); 


% %% Plot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T = 2000; %Size of time window
% s0 = 2; %Animation from time step s0
% figure('position',[10,10,2200,1200],'color',[1 1 1]); 
% 
% %Plot distribution
% subplot(3,2,1); hold on; 
% axis([xlim', -.3, max(g)*1.2]);
% xlabel('$x$','interpreter','latex','fontsize',28); 
% ylabel('$g(x)$','interpreter','latex','fontsize',28);  
% set(gca,'linewidth',2,'xtick',[],'ytick',[]);
% 
% %Plot Fourier coefficients
% subplot(3,2,2); hold on; 
% plot(rg, zeros(nbFct,1), '-','linewidth',1,'color',[0 0 0]);
% plot(rg, w_hat, '.','markersize',26,'color',[.8 0 0]);
% for n=1:nbFct
% 	plot([rg(n), rg(n)], [0, w_hat(n)], '-','linewidth',4,'color',[.8 0 0]);
% end
% axis([0, nbFct-1, min(w_hat)-5E-2, max(w_hat)+5E-2]);
% set(gca,'xtick',[0,nbFct-1],'ytick',[],'linewidth',2,'fontsize',20);
% xlabel('$k$','interpreter','latex','fontsize',28);
% ylabel('$w_k$','interpreter','latex','fontsize',28); 
% 
% %Plot signal
% subplot(3,2,[3,5]); hold on; box on;
% axis([xlim', 1, T]);
% xlabel('$x$','interpreter','latex','fontsize',28);   
% set(gca,'linewidth',2,'xtick',[],'ytick',[1,T],'yticklabel',{'t-T','t'},'fontsize',24);
% 
% %Animation
% hs = []; 
% ha = [];
% id = 1;
% for s = s0:5:nbData 
% 	%Plot distribution
% 	subplot(3,2,1); hold on; 
% 	delete(ha);
% 	ha(1) = plot(xm, g, '-','linewidth',4,'color',[.8 0 0]);
% 	ha(2) = plot(xm, r.g(:,s), '-','linewidth',2,'color',[0 0 0]);
% % 	ha(3) = plot(xm, (r.g(:,s)-r.g(:,s-1)) .* s .* 5E-2, '-','linewidth',3,'color',[0 0 .8]); %Plot increment (scaled)
% 	legend(ha,{'Desired','Reconstructed','Increment (scaled)'},'fontsize',18,'location','northwest');
% 	
% 	%Plot Fourier coefficients
% 	subplot(3,2,2); hold on; 
% 	delete(hs);
% 	hs = plot(rg, r.w(:,s), '.','markersize',18,'color',[0 0 0]);
% 	for n=1:nbFct
% 		hs = [hs, plot([rg(n), rg(n)], [0, r.w(n,s)], '-','linewidth',2,'color',[0 0 0])];
% 	end
% % 	hs = [hs, plot(rg, (r.w(:,s)-r.w(:,s-1)).*1E2, 'o','linewidth',2,'markersize',7,'color',[0 0 .8])]; %Plot increment (scaled)
% 	
% 	%Plot signal
% 	subplot(3,2,[3,5]); hold on; box on;
% 	if s>T
% 		hs = [hs, plot(r.x(:,s-T+1:s), 1:T, '-','linewidth',3,'color',[0 0 0])];
% 	else
% 		hs = [hs, plot(r.x(:,1:s), T-s+1:T, '-','linewidth',3,'color',[0 0 0])];
% 	end
% 	hs = [hs, plot(r.x(:,s), T, '.','markersize',28,'color',[0 0 0])];
% 	drawnow; 
% % 	print('-dpng',['graphs/anim/ergodicControl_1Danim' num2str(id,'%.4d') '.png']);
% % 	pause(1);
% % 	id = id + 1;
% end

pause(10);
close all;
