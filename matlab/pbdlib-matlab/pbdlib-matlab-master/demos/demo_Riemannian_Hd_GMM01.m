function demo_Riemannian_Hd_GMM01
% GMM on n-hyperboloid manifold
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
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
model.nbStates = 2; %Number of states in the GMM
model.nbVar = 3; %Dimension of the tangent space
model.nbVarMan = 3; %Dimension of the manifold
model.params_diagRegFact = 1E-2; %Regularization of covariance

nbData = 15; %Number of datapoints
nbSamples = 1; %Number of demonstrations
nbIter = 20; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 30; %Number of iteration for the EM algorithm
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids
e0 = [0; 0; 1];

x = (rand(model.nbVar,nbData)-0.5) * 1E1;
for t=1:nbData*nbSamples
	x(end,t) = (x(1:end-1,t)' * x(1:end-1,t) + 1).^.5;
	%x(1:end-1,t)' * x(1:end-1,t) - x(end,t).^2 %Should be equal to -1
end

% u = zeros(model.nbVar,nbData*nbSamples);
% for t=1:nbData*nbSamples
% 	u(:,t) = logmap(x(:,t), [0; 0; 1]);
% end
u = logmap(x, e0);


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Random initialization
% model.Priors = ones(1,model.nbStates) / model.nbStates;
% model.MuMan = expmap(randn(model.nbVar,model.nbStates), randn(model.nbVarMan,1)); %Center on the manifold
% model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
% model.Sigma = repmat(eye(model.nbVar)*5E-1,[1,1,model.nbStates]); %Covariance in the tangent plane at point MuMan

%Initialization based on kbins
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu, e0); %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x, model.MuMan(:,i)), model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
% 	%M-step
% 	for i=1:model.nbStates
% 		%Update Priors
% 		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
% 		%Update MuMan
% 		for n=1:nbIter
% 			u(:,:,i) = logmap(x, model.MuMan(:,i));
% 			model.MuMan(:,i) = expmap(u(:,:,i)*H(i,:)', model.MuMan(:,i));
% 		end
% 		%Update Sigma
% 		model.Sigma(:,:,i) = u(:,:,i) * diag(H(i,:)) * u(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
% 	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Display of covariance contours on the sphere
% tl = linspace(-pi, pi, nbDrawingSeg);
% Gdisp = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);
% for i=1:model.nbStates
% 	[V,D] = eig(model.Sigma(:,:,i));
% 	utmp = [cos(tl); sin(tl); ones(1,nbDrawingSeg)];
% % 	for t=1:nbDrawingSeg
% % 		utmp(end,t) = (utmp(1:end-1,t)' * utmp(1:end-1,t) + 1).^.5;
% % 	end
% 	Gdisp(:,:,i) = expmap(V*D.^.5*utmp, model.MuMan(:,i));
% 	for t=1:nbDrawingSeg
% 		Gdisp2(:,t,i) = Gdisp(1:end-1,t,i) / (1+Gdisp(end,t,i));
% 	end
% end
nbDrawingSeg2 = nbDrawingSeg * 10;
w = linspace(0,1,nbDrawingSeg);
wl = linspace(-12,12,nbDrawingSeg2);
clrmap = lines(model.nbStates);

figure('position',[10,10,800,800]); hold on; axis off; 
plot(0,0,'k+','linewidth',2,'markersize',20); 
if model.nbVar==3
	%Plot as Poincare disk model
	for t=1:nbData*nbSamples
		x2(:,t) = x(1:end-1,t) / (1+x(end,t));
	end
	for i=1:model.nbStates
		model.MuMan2(:,i) = model.MuMan(1:end-1,i) / (1+model.MuMan(end,i));
	end
	t = linspace(0,2*pi,100);
	plot(cos(t), sin(t), 'k-','linewidth',2);
		
% 	%Plot covariances
% 	for i=1:model.nbStates
% 		plot(Gdisp(1,:,i), Gdisp(2,:,i), 'g-');
% 		plot(Gdisp2(1,:,i), Gdisp2(2,:,i), 'r-');
% 	end
	
	for j=1:model.nbStates

% 		%Plot all geodesics prolongations
% 		for i=1:nbData*nbSamples
% 			for t=1:nbDrawingSeg
% 				xil(:,t) = expmap(wl(t)*logmap(x(:,i), model.MuMan(:,j)), model.MuMan(:,j));
% 				xil2(:,t) = xil(1:end-1,t) / (1+xil(end,t));
% 			end
% 			plot(xil2(1,:), xil2(2,:), '-','linewidth',1,'color',min(clrmap(j,:)+.2,1));
% 		end

% 		%Plot geodesics between points and all Gaussians
% 		for i=1:nbData*nbSamples
% 			for t=1:nbDrawingSeg
% 				xi(:,t) = expmap(w(t)*logmap(x(:,i), model.MuMan(:,j)), model.MuMan(:,j));
% 				xi2(:,t) = xi(1:end-1,t) / (1+xi(end,t));
% 			end
% 			plot(xi2(1,:), xi2(2,:), '-','linewidth',1,'color',clrmap(j,:));
% 		end
		
		%Plot geodesics between points and best Gaussian
		for i=1:nbData*nbSamples
			[~,id] = max(GAMMA(:,i));
			if id==j
				%Plot geodesics prolongations
				for t=1:nbDrawingSeg2
					xil(:,t) = expmap(wl(t)*logmap(x(:,i), model.MuMan(:,j)), model.MuMan(:,j));
					xil2(:,t) = xil(1:end-1,t) / (1+xil(end,t));
				end
				%plot(xil2(1,:), xil2(2,:), '-','linewidth',1,'color',min(clrmap(j,:)+.2,1));
				patch([xil2(1,:),xil2(1,end:-1:1)], [xil2(2,:),xil2(2,end:-1:1)], clrmap(j,:),'linewidth',3,'edgecolor',clrmap(j,:),'edgealpha',.05);
				
				%Plot geodesics
				for t=1:nbDrawingSeg
					xi(:,t) = expmap(w(t)*logmap(x(:,i), model.MuMan(:,j)), model.MuMan(:,j));
					xi2(:,t) = xi(1:end-1,t) / (1+xi(end,t));
				end
				%plot(xi2(1,:), xi2(2,:), '-','linewidth',1,'color',clrmap(j,:));
				patch([xi2(1,:),xi2(1,end:-1:1)], [xi2(2,:),xi2(2,end:-1:1)], clrmap(j,:),'linewidth',3,'edgecolor',clrmap(j,:),'edgealpha',.2);
			end %id
		end %i
		
		%Plot Gaussian centers
		plot(model.MuMan2(1,j), model.MuMan2(2,j), '.','markersize',40,'color',clrmap(j,:));		
	end %j
	
% 	u = logmap(x(:,2), x(:,1));
% 	u2 = u(1:end-1) / (1+u(end));
% 	plot2DArrow(x2(:,1), u2*.2, [0,0,0], 1, .05);
	plot(x2(1,:), x2(2,:), '.','markersize',30,'color',[0 0 0]);
	axis tight; axis equal; 

else
	%Plot as hyperboloid
	plot(x(1,:),x(2,:),'k.');
	u = logmap(x(:,2), x(:,1));
	plot2DArrow(x(:,1), u*.2, [0,0,0], 1, .05);
	%plot(xi(1,:),xi(2,:),'r.');
end
axis equal; 

% print('-dpng','graphs/demo_Riemannian_Hd_GMM01.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, x0)
	M = blkdiag(eye(size(x0,1)-1), -1);  
	for t=1:size(u,2)
		th = sqrt(u(:,t)' * M * u(:,t));
		if th<1e-16
			x(:,t) = x0;
		else
			%see e.g. p.224 (in pdf, p.232) of Robbin and Salamon (2013), Introduction to differential geometry
			x(:,t) = x0 .* cosh(th) + u(:,t) .* sinh(th) ./ th; 
		end
	end
end

function u = logmap(x, x0)
	M = blkdiag(eye(size(x0,1)-1), -1);  
	for t=1:size(x,2)
		
% 		theta = acosh(norm(x0'*M*x(:,t)));
% 		if theta<1e-16
% 			u(:,t) = zeros(size(x,1),1);
% 		else
% 			u(:,t) = (x(:,t) - x0 .* cosh(theta)) .* theta ./ sinh(theta);
% 		end

		e = x0' * M * x(:,t);
% 		u(:,t) = acosh(-e) .* (x(:,t) + e .* x0) ./ sqrt(e^2 - 1);
		th = acosh(-e);
		if th<1e-16
			u(:,t) = zeros(size(x,1),1);
		else
			u(:,t) = th .* (x(:,t) + e .* x0) ./ sqrt(e^2 - 1);
		end
	end
end