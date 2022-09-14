% Movement primitives applied to 2D trajectory encoding and decoding
%
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Sylvain Calinon, https://calinon.ch/
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

function demo_proMP01

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbFct = 7; %Number of basis functions (odd number for Fourier basis functions)
param.nbVar = 2; %Dimension of position data (here: x1,x2)
param.nbData = 100; %Number of datapoints in a trajectory
param.basisName = 'RBF'; %PIECEWISE, RBF, BERNSTEIN, FOURIER


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('./data/2Dletters/S.mat'); %Planar trajectories for writing alphabet letters
x = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),param.nbData))(:); %Resampling

if isequal(param.basisName,'FOURIER')
	%Fourier basis functions require additional symmetrization (mirroring) if the signal is a discrete motion 
	X = reshape(x, param.nbVar, param.nbData);
	X = [X, fliplr(X)]; %Build real and even signal, characterized by f(-x) = f(x)
	x = X(:); 
	param.nbData = param.nbData * 2;
end


%% Movement primitives encoding and decoding 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isequal(param.basisName,'PIECEWISE')
    phi = buildPhiPiecewise(param.nbData, param.nbFct);
elseif isequal(param.basisName,'RBF')
    phi = buildPhiRBF(param.nbData, param.nbFct);
elseif isequal(param.basisName,'BERNSTEIN')
    phi = buildPhiBernstein(param.nbData, param.nbFct);
elseif isequal(param.basisName,'FOURIER')
    phi = buildPhiFourier(param.nbData, param.nbFct);
end

Psi = kron(phi, eye(param.nbVar)); %Transform to multidimensional basis functions
w = (Psi' * Psi + eye(param.nbVar*param.nbFct).*realmin) \ Psi' * x; %Estimation of superposition weights from data
x_hat = Psi * w; %Reconstruction of data

if isequal(param.basisName,'FOURIER')
	%Fourier basis functions require de-symmetrization of the signal after processing (for visualization)
	param.nbData = param.nbData/2;
	x = x(1:param.nbData*param.nbVar);
	x_hat = x_hat(1:param.nbData*param.nbVar);
	phi = real(phi); %for plotting
	%Psi = Psi(1:param.nbData*param.nbVar,:);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 1800 500]); 
clrmap = lines(param.nbFct);

%Plot signals
subplot(1,3,1); hold on; axis off; 
l(1) = plot(x(1:2:end,:), x(2:2:end,:), 'linewidth',2,'color',[.2 .2 .2]);
l(2) = plot(x_hat(1:2:end,:), x_hat(2:2:end,:), '-','linewidth',2,'color',[.9 .0 .0]);
% legend(l,{'Demonstration','Reproduction'},'fontsize',20); % Throw an error with octave
axis tight; axis equal; 

%Plot basis functions (display only the real part for Fourier basis functions)
subplot(1,3,2); hold on; 
for i=1:param.nbFct
	plot(phi(:,i), 'linewidth',2,'color',clrmap(i,:));
end
if isequal(param.basisName,'FOURIER')
	plot([param.nbData,param.nbData], [-1,1], 'k:','linewidth',4);
end
axis tight;
xlabel('t','fontsize',26); ylabel('\phi_k','fontsize',26);

%Plot Psi*Psi' matrix (covariance matrix at trajectory level)
subplot(1,3,3); hold on; axis off; title('\Psi\Psi^T','fontsize',26);
msh = [0 1 1 0 0; 0 0 1 1 0];
colormap(flipud(gray));
imagesc(abs(Psi * Psi'));
plot(msh(1,:)*size(Psi,1), msh(2,:)*size(Psi,1), 'k-');
if isequal(param.basisName,'FOURIER')
	plot(msh(1,:)*size(Psi,1)/2, msh(2,:)*size(Psi,1)/2, 'k:','linewidth',4);
end
axis tight; axis square; axis ij;

waitfor(h);
end


%%Likelihood of datapoint(s) for a Gaussian parameterized with center and full covariance
%function prob = gaussPDF(Data, Mu, Sigma)    
%	[nbVar,nbData] = size(Data);
%	Data = Data - repmat(Mu,1,nbData);
%	prob = sum((Sigma\Data).*Data,1);
%	prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * abs(det(Sigma)) + realmin);
%end

%Building piecewise constant basis functions
function phi = buildPhiPiecewise(nbData, nbFct) 
%	iList = round(linspace(0, nbData, nbFct+1));
%	phi = zeros(nbData, nbFct);
%	for i=1:nbFct
%		phi(iList(i)+1:iList(i+1),i) = 1;
%	end
	phi = kron(eye(nbFct), ones(ceil(nbData/nbFct),1));
	phi = phi(1:nbData,:);
end

%Building radial basis functions (RBFs)
function phi = buildPhiRBF(nbData, nbFct) 
	t = linspace(0, 1, nbData);
	tMu = linspace(t(1), t(end), nbFct);

%	%Version 1
%	phi = zeros(nbData, nbFct);
%	for i=1:nbFct
%		phi(:,i) = gaussPDF(t, tMu(i), 1E-2);
%		%phi(:,i) = mvnpdf(t', tMu(i), 1E-2); %Requires statistics package/toolbox
%	end

%	%Version 2
%	%D = repmat(t', [1,nbFct]) - repmat(tMu, [nbData,1]);
%	D = t' * ones(1,nbFct) - ones(nbData,1) * tMu;
%	phi = exp(-1E2 * D.^2);

	%Version 3
	phi = exp(-1E1 * (t' - tMu).^2);
	
	%Optional rescaling
	%phi = phi ./ repmat(sum(phi,2), 1, nbFct); 
end

%Building Bernstein basis functions
function phi = buildPhiBernstein(nbData, nbFct)
	t = linspace(0, 1, nbData);
	phi = zeros(nbData, nbFct);
	for i=1:nbFct
		phi(:,i) = factorial(nbFct-1) ./ (factorial(i-1) .* factorial(nbFct-i)) .* (1-t).^(nbFct-i) .* t.^(i-1);
	end
end

%Building Fourier basis functions
function phi = buildPhiFourier(nbData, nbFct)
	t = linspace(0, 1, nbData);
	
%	%Computation for general signals (incl. complex numbers)
%	d = ceil((nbFct-1)/2);
%	k = -d:d;
%	phi = exp(t' * k * 2 * pi * 1i); 
%	%phi = cos(t' * k * 2 * pi); %Alternative computation for real signal
		
	%Alternative computation for real and even signal
	k = 0:nbFct-1;
	phi = cos(t' * k * 2 * pi);
	%phi(:,2:end) = phi(:,2:end) * 2;
	%invPhi = cos(k' * t * 2 * pi) / nbData;
	%invPsi = kron(invPhi, eye(param.nbVar));
end
