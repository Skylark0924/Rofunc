%%    Linear quadratic tracking (LQT) with control primitives applied to a viapoint task (batch formulation)
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Sylvain Calinon <https://calinon.ch>
%%
%%    This file is part of RCFS.
%%
%%    RCFS is free software: you can redistribute it and/or modify
%%    it under the terms of the GNU General Public License version 3 as
%%    published by the Free Software Foundation.
%%
%%    RCFS is distributed in the hope that it will be useful,
%%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%%    GNU General Public License for more details.
%%
%%    You should have received a copy of the GNU General Public License
%%    along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function LQT_CP

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 100; %Number of datapoints
param.nbPoints = 3; %Number of viapoints
param.nbVarPos = 2; %Dimension of position data (here: x1,x2)
param.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarPos * param.nbDeriv; %Dimension of state vector
param.dt = 1E-2; %Time step duration
param.nbFct = 12; %Number of basis functions
param.basisName = 'RBF'; %PIECEWISE, RBF, BERNSTEIN, FOURIER
param.r = 1E-8; %Control cost in LQR 

%Task setting (viapoints passing)
Mu = [rand(param.nbVarPos, param.nbPoints) - 0.5; zeros(param.nbVar-param.nbVarPos, param.nbPoints)]; %Viapoints
%Q = speye(param.nbVar * param.nbPoints) * 1E0; %Precision matrix (for full state)
Q = kron(eye(param.nbPoints), diag([ones(1,param.nbVarPos) * 1E0, zeros(1,param.nbVar-param.nbVarPos)])); %Precision matrix (for position only)
R = speye((param.nbData-1)*param.nbVarPos) * param.r; %Standard control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVar + [1:param.nbVar]';

%Dynamical System settings (discrete version)
A1d = zeros(param.nbDeriv);
for i=0:param.nbDeriv-1
	A1d = A1d + diag(ones(param.nbDeriv-i,1),i) * param.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(param.nbDeriv,1); 
for i=1:param.nbDeriv
	B1d(param.nbDeriv-i+1) = param.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, speye(param.nbVarPos)); %Discrete nD
B = kron(B1d, speye(param.nbVarPos)); %Discrete nD

%Build Sx and Su transfer matrices
Sx0 = kron(ones(param.nbData,1), speye(param.nbVar));
Su0 = sparse(param.nbVar*param.nbData, param.nbVarPos*(param.nbData-1));
M = B;
for n=2:param.nbData
	id1 = (n-1)*param.nbVar+1:param.nbData*param.nbVar;
	Sx0(id1,:) = Sx0(id1,:) * A;
	id1 = (n-1)*param.nbVar+1:n*param.nbVar; 
	id2 = 1:(n-1)*param.nbVarPos;
	Su0(id1,id2) = M;
	M = [A*M(:,1:param.nbVarPos), M]; 
end
Su = Su0(idx,:);
Sx = Sx0(idx,:);

%Build basis functions
if isequal(param.basisName,'PIECEWISE')
	phi = buildPhiPiecewise(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'RBF')
	phi = buildPhiRBF(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'BERNSTEIN')
	phi = buildPhiBernstein(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'FOURIER')
	phi = buildPhiFourier(param.nbData-1, param.nbFct);
end

%Application of basis functions to multidimensional control commands
Psi = kron(phi, eye(param.nbVarPos)); 


%% Linear quadratic tracking (batch formulation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = zeros(param.nbVar,1); %Initial state
w = (Psi' * Su' * Q * Su * Psi + Psi' * R * Psi) \ Psi' * Su' * Q * (Mu(:) - Sx * x0); %Superposition weights
u = Psi * w; %Control commands
rx = reshape(Sx0*x0+Su0*u, param.nbVar, param.nbData); %Reproduced trajectory
ru = reshape(u, param.nbVarPos, param.nbData-1);    

%Post-processing for plotting
if isequal(param.basisName,'FOURIER')
	rx = real(rx);
	ru = real(ru);
	phi = real(phi); %for plotting
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h(1) = figure('position',[10 10 800 800]); hold on; axis off;
plot(rx(1,:), rx(2,:), 'k-','linewidth',2);
plot(rx(1,1), rx(2,1), 'k.','markersize',30);
plot(Mu(1,:), Mu(2,:), 'r.','markersize',30);
axis equal; 


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(param.nbFct);
h(2) = figure('position',[950 10 800 800]); 
%States plot
for j=1:param.nbVarPos
	subplot(2*param.nbVarPos+1,1,j); hold on;
	plot(rx(j,:), 'k-','linewidth',2);
	plot(rx(j,1), 'k.','markersize',30);
	plot(tl, Mu(j,:), 'r.','markersize',30);
	ylabel(['x_' num2str(j)], 'fontsize',26);
end
%Commands plot
for j=1:param.nbVarPos
	subplot(2*param.nbVarPos+1,1,param.nbVarPos+j); hold on;
	plot(ru(j,:), 'k-','linewidth',2);
	ylabel(['u_' num2str(j)], 'fontsize',26);
end
%Basis functions plot
subplot(2*param.nbVarPos+1,1,2*param.nbVarPos+1); hold on; 
for i=1:param.nbFct
	plot(phi(:,i), '-','linewidth',3,'color',clrmap(i,:));
end
xlabel('t','fontsize',26); 
ylabel('\phi_k','fontsize',26);

waitfor(h)
end


%Building piecewise constant basis functions
function phi = buildPhiPiecewise(nbData, nbFct) 
	phi = kron(eye(nbFct), ones(ceil(nbData/nbFct),1));
	phi = phi(1:nbData,:);
end

%Building radial basis functions (RBFs)
function phi = buildPhiRBF(nbData, nbFct) 
	t = linspace(0, 1, nbData);
	tMu = linspace(t(1)-1/(nbFct-3), t(end)+1/(nbFct-3), nbFct); %Repartition of centers to limit border effects
	sigma = 1 / (nbFct-2); %Standard deviation
	phi = exp(-(t' - tMu).^2 / sigma^2);
	
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
	
	%Computation for general signals (incl. complex numbers)
	d = ceil((nbFct-1)/2);
	k = -d:d;
	phi = exp(t' * k * 2 * pi * 1i); 
	%phi = cos(t' * k * 2 * pi); %Alternative computation for real signal
		
%	%Alternative computation for real and even signal
%	k = 0:nbFct-1;
%	phi = cos(t' * k * 2 * pi);
%	%phi(:,2:end) = phi(:,2:end) * 2;
%	%invPhi = cos(k' * t * 2 * pi) / nbData;
%	%invPsi = kron(invPhi, eye(param.nbVar));
end
