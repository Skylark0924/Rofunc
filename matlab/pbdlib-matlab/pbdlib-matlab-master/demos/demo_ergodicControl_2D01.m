function demo_ergodicControl_2D01
% 2D ergodic control with a spatial distribution described as a GMM, inspired by G. Mathew and I. Mezic, 
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
nbData = 2000; %Number of datapoints
nbFct = 10; %Number of basis functions along x and y
nbVar = 2; %Dimension of datapoint
nbStates = 2; %Number of Gaussians to represent the spatial distribution
sp = (nbVar + 1) / 2; %Sobolev norm parameter
dt = 1E-2; %Time step
xlim = [0; 1]; %Domain limit for each dimension (considered to be 1 for each dimension in this implementation)
L = (xlim(2) - xlim(1)) * 2; %Size of [-xlim(2),xlim(2)]
om = 2 * pi / L; %omega parameter
u_max = 1E1; %Maximum speed allowed 

%Desired spatial distribution represented as a mixture of Gaussians
% Mu(:,1) = [.3; .7]; 
% Sigma(:,:,1) = eye(nbVar).*1E-2; 
% Mu(:,2) =  [.7; .4]; 
% Sigma(:,:,2) = [.1;.2]*[.1;.2]' .*8E-1 + eye(nbVar)*1E-3; 

Mu(:,1) = [.5; .7]; 
Sigma(:,:,1) = [.3;.1]*[.3;.1]' *5E-1 + eye(nbVar)*5E-3; %eye(nbVar).*1E-2; 
Mu(:,2) =  [.6; .3]; 
Sigma(:,:,2) = [.1;.2]*[.1;.2]' *3E-1 + eye(nbVar)*1E-2;

% Mu = rand(nbVar, nbStates);
% Sigma = repmat(eye(nbVar)*1E-1, [1,1,nbStates]);

Priors = ones(1,nbStates) / nbStates; %Mixing coefficients


%% Compute Fourier series coefficients phi_k of desired spatial distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[xx, yy] = ndgrid(1:nbFct, 1:nbFct);
rg = 0:nbFct-1;
[KX(1,:,:), KX(2,:,:)] = ndgrid(rg, rg);
Lambda = (KX(1,:).^2 + KX(2,:).^2 + 1)'.^-sp; %Weighting vector (Eq.(15))

%Explicit description of phi_k by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries),
%by enumerating symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1]), and removing redundant ones -> keeping ([-1,-1],[-1,1])
op = hadamard(2^(nbVar-1));
op = op(1:nbVar,:);
%Compute phi_k
kk = KX(:,:) * om;
w_hat = zeros(nbFct^nbVar, 1);
for j=1:nbStates
	for n=1:size(op,2)
		MuTmp = diag(op(:,n)) * Mu(:,j); %Eq.(20)
		SigmaTmp = diag(op(:,n)) * Sigma(:,:,j) * diag(op(:,n))'; %Eq.(20)
		w_hat = w_hat + Priors(j) * cos(kk' * MuTmp) .* exp(diag(-.5 * kk' * SigmaTmp * kk)); %Eq.(21)
	end
end
w_hat = w_hat / L^nbVar / size(op,2);


% %Alternative computation of w_hat by discretization (for verification)
% nbRes = 100;
% xm1d = linspace(xlim(1), xlim(2), nbRes); %Spatial range for 1D
% [xm(1,:,:), xm(2,:,:)] = ndgrid(xm1d, xm1d); %Spatial range
% g = zeros(1,nbRes^nbVar);
% for k=1:nbStates
% 	g = g + Priors(k) * mvnpdf(xm(:,:)', Mu(:,k)', Sigma(:,:,k))'; %Spatial distribution
% end
% phi_inv = cos(KX(1,:)' * xm(1,:) * om) .* cos(KX(2,:)' * xm(2,:) * om) / L^nbVar / nbRes^nbVar;
% w_hat = phi_inv * g'; %Fourier coefficients of spatial distribution


%Fourier basis functions (for a discretized map)
nbRes = 100;
xm1d = linspace(xlim(1), xlim(2), nbRes); %Spatial range for 1D
[xm(1,:,:), xm(2,:,:)] = ndgrid(xm1d, xm1d); %Spatial range
phim = cos(KX(1,:)' * xm(1,:) .* om) .* cos(KX(2,:)' * xm(2,:) * om) * 2^nbVar; %Fourier basis functions
% % phim(2:end,:) = phim(2:end,:) * 2;
% phim = phim .* 2^nbVar;
% phim(1:nbFct,:) = phim(1:nbFct,:) * .5;
% phim(1:nbFct:end,:) = phim(1:nbFct:end,:) * .5;
hk = [1; 2*ones(nbFct-1,1)];
HK = hk(xx(:)) .* hk(yy(:)); 
phim = phim .* repmat(HK,[1,nbRes^nbVar]);

%Desired spatial distribution 
g = w_hat' * phim;

% %Alternative computation of g
% g = zeros(1,nbRes^nbVar);
% for k=1:nbStates
% 	g = g + Priors(k) * mvnpdf(xm(:,:)', Mu(:,k)', Sigma(:,:,k))'; %Spatial distribution
% end


%% Ergodic control 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [.1; .3]; %Initial position
wt = zeros(nbFct^nbVar, 1);
for t=1:nbData
	r.x(:,t) = x; %Log data
	
	%Fourier basis functions and derivatives for each dimension (only cosine part on [0,L/2] is computed since the signal is even and real by construction) 
	phi1 = cos(x * rg * om); %Eq.(18)
	dphi1 = -sin(x * rg * om) .* repmat(rg,nbVar,1) * om;
	
	dphi = [dphi1(1,xx) .* phi1(2,yy); phi1(1,xx) .* dphi1(2,yy)]; %Gradient of basis functions
	wt = wt + (phi1(1,xx) .* phi1(2,yy))' / L^nbVar;	%wt./t are the Fourier series coefficients along trajectory (Eq.(17))

% 	%Controller with ridge regression formulation
% 	u = -dphi * (Lambda .* (wt./t - w_hat)) .* t .* 5E-1; %Velocity command
	
	%Controller with constrained velocity norm
	u = -dphi * (Lambda .* (wt/t - w_hat)); %Eq.(24)
	u = u * u_max / (norm(u)+1E-1); %Velocity command
	
	x = x + u * dt; %Update of position
	r.g(:,t) = (wt/t)' * phim; %Reconstructed spatial distribution (for visualization)
	r.w(:,t) = wt/t; %Fourier coefficients along trajectory (for visualization)
% 	r.e(t) = sum(sum((wt./t - w_hat).^2 .* Lambda)); %Reconstruction error evaluation
end


% %The Fourier series coefficients along trajectory can alternatively be computed in batch form
% phi1 = [];
% size(cos(r.x(1:2:end)' * rg .* om))
% phi1(1,:,:) = cos(r.x(1:2:end)' * rg .* om);
% phi1(2,:,:) = cos(r.x(2:2:end)' * rg .* om);
% wt2 = sum( (phi1(1,:,xx) .* phi1(2,:,yy)) ) ./ L.^nbVar;
% norm(wt-wt2(:))
% return


%% Plot (static)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1400,800]); 
colormap(repmat(linspace(1,.4,64),3,1)');

%x
subplot(1,3,1); hold on; axis off;
G = reshape(g,[nbRes,nbRes]);
% G = reshape(r.g(:,end),[nbRes,nbRes]);
G([1,end],:) = max(g); %Add vertical image borders
G(:,[1,end]) = max(g); %Add horizontal image borders
surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), G, 'FaceColor','interp','EdgeColor','interp');
% surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), reshape(r.g(:,end),[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp');
% plotGMM(Mu, Sigma, [.2 .2 .2], .3);
plot(r.x(1,:), r.x(2,:), '-','linewidth',1,'color',[0 0 0]);
% plot(r.x(1,:), r.x(2,:), '.','markersize',10,'color',[0 0 0]);
plot(r.x(1,1), r.x(2,1), '.','markersize',15,'color',[0 0 0]);
axis([xlim(1),xlim(2),xlim(1),xlim(2)]); axis equal;

%w
subplot(1,3,2); hold on; axis off; title('$w$','interpreter','latex','fontsize',20);
imagesc(reshape(wt./t,[nbFct,nbFct]));
axis tight; axis equal; axis ij;

%w_hat
subplot(1,3,3); hold on; axis off; title('$\hat{w}$','interpreter','latex','fontsize',20);
imagesc(reshape(w_hat,nbFct,nbFct));
axis tight; axis equal; axis ij;
% print('-dpng','graphs/ergodicControl_2D01.png'); 


% %% Additional plots (static)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %Plot g as image
% % figure; hold on;
% % imagesc(reshape(g,[nbRes,nbRes])');
% % axis tight; axis equal; %axis ij;
% 
% % %Plot g as graph
% % figure('position',[10,10,2300,1300]); 
% % subplot(1,2,1); hold on; rotate3d on;
% % surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), reshape(g,[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp');
% % view(3); axis vis3d;
% % subplot(1,2,2); hold on; rotate3d on;
% % surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), reshape(r.g(:,end),[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp');
% % view(3); axis vis3d;


% %% Plot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,2200,800]); 
% %x
% subplot(1,3,1); hold on; axis off;
% colormap(repmat(linspace(1,.4,64),3,1)');
% G = reshape(g,[nbRes,nbRes]);
% G([1,end],:) = max(g);
% G(:,[1,end]) = max(g);
% surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), G, 'FaceColor','interp','EdgeColor','interp');
% plot(r.x(1,1), r.x(2,1), '.','markersize',15,'color',[0 0 0]);
% axis([xlim(1),xlim(2),xlim(1),xlim(2)]); axis equal;
% %w
% subplot(1,3,2); hold on; axis off; title('$w$','interpreter','latex','fontsize',20);
% colormap(repmat(linspace(1,.4,64),3,1)');
% imagesc(reshape(w_hat,nbFct,nbFct));
% % surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), reshape((r.g(:,end)-r.g(:,end-1)).*1E2,[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp'); %Plot increment
% axis tight; axis equal; axis ij;
% %w_hat
% subplot(1,3,3); hold on; axis off; title('$\hat{w}$','interpreter','latex','fontsize',20);
% colormap(repmat(linspace(1,.4,64),3,1)');
% imagesc(reshape(w_hat,nbFct,nbFct));
% axis tight; axis equal; axis ij;
% %Animation
% h = []; 
% % id = 1;
% for t=2:50:nbData
% 	delete(h);
% 	h=[];
% 	%x
% 	subplot(1,3,1); hold on; axis off; 
% % 	h = [h, surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), reshape(r.g(:,t),[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp')];
% 	h = [h, plot(r.x(1,1:t), r.x(2,1:t), '-','linewidth',1,'color',[0 0 0])];
% 	%w
% 	subplot(1,3,2); hold on; axis off;
% 	h = [h, imagesc(reshape(r.w(:,t),nbFct,nbFct))];
% % 	h = [h, imagesc(reshape((r.w(:,t)-r.w(:,t-1)).*1E2, nbFct, nbFct))]; %Plot increment (scaled)
% % 	h = h, surface(squeeze(xm(1,:,:)), squeeze(xm(2,:,:)), zeros([nbRes,nbRes]), reshape((r.g(:,t)-r.g(:,t-1)).*1E2,[nbRes,nbRes]), 'FaceColor','interp','EdgeColor','interp')]; %Plot increment
% 	drawnow;
% % 	print('-dpng',['graphs/anim/ergodicControl_2Danim' num2str(id,'%.4d') '.png']);
% % 	pause(.5);
% % 	[t id]
% % 	id = id + 1;
% end

pause;
close all;