function demo_ergodicControl_3D01
% 3D ergodic control with a spatial distribution described as a GMM, inspired by G. Mathew and I. Mezic, 
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
nbVar = 3; %Dimension of datapoint
nbStates = 8; %Number of Gaussians to represent the spatial distribution
sp = (nbVar + 1) / 2; %Sobolev norm parameter
dt = 1E-2; %Time step
xlim = [0; 1]; %Domain limit for each dimension (considered to be 1 for each dimension in this implementation)
L = (xlim(2) - xlim(1)) * 2; %Size of [-xlim(2),xlim(2)]
om = 2 * pi / L; %omega
u_max = 1E1; %Maximum speed allowed 

%Desired spatial distribution represented as a mixture of Gaussians
% Mu(:,1) = [.4; .5; .7]; 
% Sigma(:,:,1) = eye(nbVar).*1E-2; 
% Mu(:,2) =  [.7; .4; .3]; 
% Sigma(:,:,2) = [.1;.2;.3]*[.1;.2;.3]' .*8E-1 + eye(nbVar)*1E-3; 
Mu = rand(nbVar, nbStates);
Sigma = repmat(eye(nbVar)*1E-1, [1,1,nbStates]);
Priors = ones(1,nbStates) ./ nbStates; %Mixing coefficients


%% Compute Fourier series coefficients phi_k of desired spatial distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[xx, yy, zz] = ndgrid(1:nbFct, 1:nbFct, 1:nbFct);
rg = 0:nbFct-1;
[KX(1,:,:,:), KX(2,:,:,:), KX(3,:,:,:)] = ndgrid(rg, rg, rg);
Lambda = (KX(1,:).^2 + KX(2,:).^2 + KX(3,:).^2 + 1)'.^-sp; %Weighting vector (Eq.(15))

% HK = L^nbVar; %Rescaling term (as scalar)
% hk = [1; sqrt(.5)*ones(nbFct-1,1)];
% HK = hk(xx,1) .* hk(yy,1) .* hk(zz,1); %Rescaling term (as normalizing matrix)

% tic
%Explicit description of phi_k by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries)
%Enumerate symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1]), and removing redundant ones -> keeping ([-1,-1],[-1,1])
op = hadamard(2^(nbVar-1));
op = op(1:nbVar,:);
%Compute phi_k
kk = KX(:,:) * om;
w_hat = zeros(nbFct^nbVar, 1);
for j=1:nbStates
	for n=1:size(op,2)
		MuTmp = diag(op(:,n)) * Mu(:,j); %Eq.(20)
		SigmaTmp = diag(op(:,n)) * Sigma(:,:,j) * diag(op(:,n))'; %Eq.(20)
		w_hat = w_hat + Priors(j) .* cos(kk' * MuTmp) .* exp(diag(-.5 * kk' * SigmaTmp * kk)); %Eq.(21)
	end
end
w_hat = w_hat / L^nbVar / size(op,2);
% toc
% return

% %Alternative computation of w_hat by discretization (for verification)
% nbRes = 100;
% xm1d = linspace(xlim(1), xlim(2), nbRes); %Spatial range for 1D
% [xm(1,:,:,:), xm(2,:,:,:), xm(3,:,:,:)] = ndgrid(xm1d, xm1d, xm1d); %Spatial range
% g = zeros(1,nbRes^nbVar);
% for k=1:nbStates
% 	g = g + Priors(k) .* mvnpdf(xm(:,:)', Mu(:,k)', Sigma(:,:,k))'; %Spatial distribution
% end
% phi_inv = cos(KX(1,:)' * xm(1,:) .* om) .* cos(KX(2,:)' * xm(2,:) .* om) .* cos(KX(3,:)' * xm(3,:) .* om) ./ L^nbVar ./ nbRes^nbVar;
% % phi_inv = ones(nbFct^nbVar, nbRes^nbVar);
% % for n=1:nbVar
% % 	phi_inv = phi_inv .* cos(KX(n,:)' * xm(n,:) .* om) ./ L^nbVar ./ nbRes^nbVar;
% % end
% w_hat = phi_inv * g'; %Fourier coefficients of spatial distribution


%Fourier basis functions (for a discretized map)
nbRes = 20;
xm1d = linspace(xlim(1), xlim(2), nbRes); %Spatial range for 1D
[xm(1,:,:,:), xm(2,:,:,:), xm(3,:,:,:)] = ndgrid(xm1d, xm1d, xm1d); %Spatial range
phim = cos(KX(1,:)' * xm(1,:) .* om) .* cos(KX(2,:)' * xm(2,:) .* om) .* cos(KX(3,:)' * xm(3,:) .* om) .* 2^nbVar; %Fourier basis functions
hk = [1; 2*ones(nbFct-1,1)];
HK = hk(xx(:)) .* hk(yy(:)) .* hk(zz(:)); 
phim = phim .* repmat(HK,[1,nbRes^nbVar]);

%Desired spatial distribution 
g = w_hat' * phim;

% %Alternative computation of g
% g = zeros(1,nbRes^nbVar);
% for k=1:nbStates
% 	g = g + Priors(k) .* mvnpdf(xm(:,:)', Mu(:,k)', Sigma(:,:,k))'; %Spatial distribution
% end


%% Ergodic control 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [.7; .1; .5]; %Initial position
wt = zeros(nbFct^nbVar, 1);
for t=1:nbData
	r.x(:,t) = x; %Log data

	%Fourier basis functions and derivatives for each dimension (only cosine part on [0,L/2] is computed since the signal is even and real by construction) 
	phi1 = cos(x * rg .* om); %Eq.(18)
	dphi1 = -sin(x * rg .* om) .* repmat(rg,nbVar,1) .* om;
	
	dphi = [dphi1(1,xx) .* phi1(2,yy) .* phi1(3,zz); phi1(1,xx) .* dphi1(2,yy) .* phi1(3,zz); phi1(1,xx) .* phi1(2,yy) .* dphi1(3,zz)]; %Gradient of basis functions
	wt = wt + (phi1(1,xx) .* phi1(2,yy) .* phi1(3,zz))' ./ L^nbVar;	%wt./t are the Fourier series coefficients along trajectory (Eq.(17))
	
% 	%Controller with ridge regression formulation
% 	u = -dphi * (Lambda .* (wt./t - w_hat)) .* t .* 1E-1; %Velocity command
	
	%Controller with constrained velocity norm
	u = -dphi * (Lambda .* (wt./t - w_hat)); %Eq.(24)
	u = u .* u_max ./ (norm(u)+1E-1); %Velocity command
	
	x = x + u .* dt; %Update of position
	
% 	r.g(:,t) = (wt./t)' * phim; %Reconstructed spatial distribution 
% 	r.e(t) = sum(sum((wt./t - w_hat).^2 .* Lambda)); %Reconstruction error evaluation
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800]); hold on; axis on; box on; rotate3d on;
% colormap(repmat(linspace(1,0,64),3,1)');
colormap hsv;

plotGMM3D(Mu, Sigma, [.2 .2 .2], .3, 2);

% [x,y,z] = meshgrid(xm1d, xm1d, xm1d);
% v = reshape(g,[nbRes,nbRes,nbRes]);
% v = permute(v,[2,1,3]);
% v(v<1E-3) = nan; 
% % pcolor3(x,y,z,v,'alpha',.1);
% % vol3d('XData',x,'YData',y,'ZData',z,'CData',v);
% h = slice(x, y, z, v, [], [], xlim(1):.02:xlim(2));
% set(h,'edgecolor','none','facecolor','interp','facealpha','interp');
% alpha('color');
% % alphamap('rampdown')
% % alphamap('increase',.05)

plot3(Mu(1,:), Mu(2,:), Mu(3,:), '.','markersize',15,'color',[0 0 0]);
plot3(r.x(1,:), r.x(2,:), r.x(3,:), '-','linewidth',1,'color',[0 0 0]);
plot3(r.x(1,1), r.x(2,1), r.x(3,1), '.','markersize',15,'color',[0 0 0]);
plot3Dframe(eye(3).*.3, ones(3,1).*.2);
axis([xlim(1),xlim(2),xlim(1),xlim(2),xlim(1),xlim(2)]); axis equal; axis vis3d; view(50,20);
set(gca,'xtick',[],'ytick',[],'ztick',[]);
% print('-dpng','graphs/ergodicControl_3D01.png'); 

pause;
close all;