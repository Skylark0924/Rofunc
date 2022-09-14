function demo_ergodicControl_nD01
% nD ergodic control with a spatial distribution described as a GMM, inspired by G. Mathew and I. Mezic, 
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
nbFct = 5; %Number of basis functions along x and y
nbVar = 6; %Dimension of datapoint
nbStates = 2; %Number of Gaussians to represent the spatial distribution
sp = (nbVar + 1) / 2; %Sobolev norm parameter
dt = 1E-2; %Time step
xlim = [0; 1]; %Domain limit for each dimension (considered to be 1 for each dimension in this implementation)
L = (xlim(2) - xlim(1)) * 2; %Size of [-xlim(2),xlim(2)]
om = 2 * pi / L; %omega
u_max = 5E1; %Maximum speed allowed 

%Desired spatial distribution represented as a mixture of Gaussians
Mu = rand(nbVar, nbStates);
Sigma = zeros(nbVar, nbVar, nbStates);
for n=1:nbStates
	Sigma(:,:,n) = cov(rand(10,nbVar));
end
% Mu = rand(nbVar, nbStates);
% Sigma = repmat(eye(nbVar)*1E-1, [1,1,nbStates]);
Priors = ones(1,nbStates) ./ nbStates; %Mixing coefficients


%% Compute Fourier series coefficients phi_k of desired spatial distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
arr = ndarr(1:nbFct, nbVar);
rg = 0:nbFct-1;
Karr = ndarr(rg, nbVar);
stmp = zeros(nbFct^nbVar, 1);
for n=1:nbVar
	stmp = stmp + Karr(n).x(:).^2;
end
Lambda = (stmp + 1).^-sp; %Weighting vector (Eq.(15))

% HK = L^nbVar; %Rescaling term (as scalar)
% % hk = [1; sqrt(.5)*ones(nbFct-1,1)];
% % HK = ones(nbFct^nbVar, 1);
% % for n=1:nbVar
% % 	HK = HK .* hk(arr(n).x(:),1); %Rescaling term (as normalizing matrix)
% % end

tic
%Explicit description of phi_k by exploiting the Fourier transform properties of Gaussians (optimized version by exploiting symmetries)
%Enumerate symmetry operations for 2D signal ([-1,-1],[-1,1],[1,-1] and [1,1]), and removing redundant ones -> keeping ([-1,-1],[-1,1])
op = hadamard(2^(nbVar-1));
op = op(1:nbVar,:);
%Compute phi_k
kk = [];
for n=1:nbVar
	kk = [kk; Karr(n).x(:)' * om];
end
w_hat = zeros(nbFct^nbVar, 1);
for j=1:nbStates
	for n=1:size(op,2)
		MuTmp = diag(op(:,n)) * Mu(:,j); %Eq.(20)
		SigmaTmp = diag(op(:,n)) * Sigma(:,:,j) * diag(op(:,n))'; %Eq.(20)
		w_hat = w_hat + Priors(j) .* cos(kk' * MuTmp) .* exp(diag(-.5 * kk' * SigmaTmp * kk)); %Eq.(21)
	end
end
w_hat = w_hat / L^nbVar / size(op,2);
toc

% %Alternative computation of w_hat by discretization (for verification)
% nbRes = 10;
% xm1d = linspace(xlim(1), xlim(2), nbRes); %Spatial range for 1D
% [KX(1,:,:,:,:), KX(2,:,:,:,:), KX(3,:,:,:,:), KX(4,:,:,:,:)] = ndgrid(rg, rg, rg, rg); %for nbVar=4
% [xm(1,:,:,:,:), xm(2,:,:,:,:), xm(3,:,:,:,:), xm(4,:,:,:,:)] = ndgrid(xm1d, xm1d, xm1d, xm1d); %Spatial range (for nbVar=4)
% xmarr = ndarr(xm1d, nbVar); %Spatial range (for nbVar>4)
% g = zeros(1, nbRes^nbVar);
% for k=1:nbStates
% 	g = g + Priors(k) .* mvnpdf(xm(:,:)', Mu(:,k)', Sigma(:,:,k))'; %Spatial distribution
% end
% % phi_inv = cos(KX(1,:)' * xm(1,:) .* om) .* cos(KX(2,:)' * xm(2,:) .* om)
% % .* cos(KX(3,:)' * xm(3,:) .* om) .* cos(KX(4,:)' * xm(4,:) .* om) ./ L^nbVar ./ nbRes^nbVar; %for nbVar=4
% phi_inv = ones(nbFct^nbVar, nbRes^nbVar);
% for n=1:nbVar
% 	phi_inv = phi_inv .* cos(KX(n,:)' * xm(n,:) .* om) ./ L^nbVar ./ nbRes^nbVar; %for nbVar=4
% % 	phi_inv = phi_inv .* cos(Karr(n).x(:) * xmarr(n).x(:)' .* om) ./ L^nbVar ./ nbRes^nbVar; %for nbVar>4
% end
% w_hat = phi_inv * g'; %Fourier coefficients of spatial distribution

tic
%% Ergodic control 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = rand(nbVar,1); %Initial position
wt = zeros(nbFct^nbVar, 1);
for t=1:nbData
	r.x(:,t) = x; %Log data
	
	%Fourier basis functions and derivatives for each dimension (only cosine part on [0,L/2] is computed since the signal is even and real by construction) 
	phi1 = cos(x * rg .* om); %Eq.(18)
	dphi1 = -sin(x * rg .* om) .* repmat(rg,nbVar,1) .* om;
	
	dphi = ones(nbVar, nbFct^nbVar);
	phi = ones(nbFct^nbVar, 1);
	for n=1:nbVar
		for m=1:nbVar
			if m==n
				dphi(n,:) = dphi(n,:) .* dphi1(m,arr(m).x(:));
			else
				dphi(n,:) = dphi(n,:) .* phi1(m,arr(m).x(:));
			end
		end
		phi = phi .* phi1(n,arr(n).x(:))';
	end	
	wt = wt + phi ./ L^nbVar;	%wt./t are the Fourier series coefficients along trajectory (Eq.(17))
	
% 	%Controller with ridge regression formulation
% 	u = -dphi * (Lambda .* (wt./t - w_hat)) .* t .* 1E-1; %Velocity command
	
	%Controller with constrained velocity norm
	u = -dphi * (Lambda .* (wt./t - w_hat)); %Eq.(24)
	u = u .* u_max ./ (norm(u)+1E-1); %Velocity command
	
	x = x + u .* dt; %Update of position
end
toc


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1200,800]); hold on; axis off; rotate3d on;
plotGMM3D(Mu(1:3,:), Sigma(1:3,1:3,:), [.2 .2 .2], .3, 2);
plot3(Mu(1,:), Mu(2,:), Mu(3,:), '.','markersize',15,'color',[0 0 0]);
plot3(r.x(1,:), r.x(2,:), r.x(3,:), '-','linewidth',1,'color',[0 0 0]);
plot3(r.x(1,1), r.x(2,1), r.x(3,1), '.','markersize',15,'color',[0 0 0]);
axis([xlim(1),xlim(2),xlim(1),xlim(2),xlim(1),xlim(2)]); axis equal; axis vis3d; view(60,25);
% print('-dpng','graphs/ergodicControl_nD01.png'); 

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function arr = ndarr(lst, nbVar)
% 	x = [];
	for n=1:nbVar
		s = ones(1,nbVar); 
		s(n) = numel(lst);
		lst = reshape(lst,s);
		s = repmat(numel(lst),1,nbVar); 
		s(n) = 1;
% 		x = cat(n+1,x,repmat(lst,s));
		arr(n).x = repmat(lst,s);
	end
end