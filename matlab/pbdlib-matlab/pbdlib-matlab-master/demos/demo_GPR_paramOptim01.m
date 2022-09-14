function demo_GPR_paramOptim01
% Learning of the kernel parameters for Gaussian process regression
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Noémie Jaquier, http://njaquier.ch/
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
nbData = 100; %Length of each trajectory
nbSamples = 50;
dt = 0.01;

%% Generate trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DataIn = linspace(0,nbData*dt,nbData);
% Define kernel parameters
gamma = [10; 10; 1E-2];

% Compute the kernel
r2 = pdist2(DataIn',DataIn').^2;
k = gamma(1)*gamma(1) .* exp(-r2.*gamma(2).*gamma(2)) + gamma(3).*gamma(3).*eye(nbData); 

% Generate stochastic samples from the prior 
Data = zeros(2,nbData*nbSamples);
Data(1,:) = repmat(DataIn,1,nbSamples);

[V,D] = eig(k);
for s=1:nbSamples
	Data(2,(s-1)*nbData+1:s*nbData) = (real(V*D^.5) * randn(nbData,1))'; 
end

fprintf('Parameters: %.3f, %.3f, %.3f\n',gamma(1),1/gamma(2),gamma(3));

%% Learn the parameters from the generated data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Both parametrization guarantee that the final parameters of the kernel
% are positive.
% Remark: the initial estimate is important to ensure the convergence to
% correct parameters.

% Parametrization with log such that: k = exp(g2) * exp(-d*d*exp(g1)) + exp(g3)*I
gamma0 = [1;2;-2];
LLfct =@(gamma) LogLikelihood_logparam(Data, gamma, nbData, nbSamples);
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
problem.options = options;
problem.x0 = gamma0;
problem.objective = LLfct;
problem.solver = 'fminunc';
log_gamma_hat = fminunc(problem);
gamma_hat1 = sqrt(exp(log_gamma_hat));

fprintf('Learned parameters 1: %.3f, %.3f, %.3f\n',gamma_hat1(1),1/gamma_hat1(2),gamma_hat1(3));
fprintf('Initial LL %.3f, final LL %.3f \n', -LLfct(gamma0), -LLfct(log_gamma_hat));

% Parametrization such that: k = g2*g2 * exp(-d*d*g1*g1) + g3*g3*I
gamma0 = [3;3;1E-3];
LLfct =@(gamma) LogLikelihood(Data, gamma, nbData, nbSamples);
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
problem.options = options;
problem.x0 = gamma0;
problem.objective = LLfct;
problem.solver = 'fminunc';
gamma_hat2 = fminunc(problem);

fprintf('Learned parameters 1: %.3f, %.3f, %.3f\n',gamma_hat2(1),1/gamma_hat2(2),gamma_hat2(3));
fprintf('Initial LL %.3f, final LL %.3f\n', -LLfct(gamma0), -LLfct(gamma_hat2));

%% Regenerate trajectories from learned parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = gamma_hat1(1)*gamma_hat1(1) .* exp(-r2.*gamma_hat1(2).*gamma_hat1(2)) + gamma_hat1(3).*gamma_hat1(3).*eye(nbData); 
% Generate stochastic samples from the prior 
[V,D] = eig(k);
for s=1:nbSamples
	Data_hat1(1,(s-1)*nbData+1:s*nbData) = (real(V*D^.5) * randn(nbData,1))'; 
end

k = gamma_hat2(1)*gamma_hat2(1) .* exp(-r2.*gamma_hat2(2).*gamma_hat2(2)) + gamma_hat2(3).*gamma_hat2(3).*eye(nbData); 
% Generate stochastic samples from the prior 
[V,D] = eig(k);
for s=1:nbSamples
	Data_hat2(1,(s-1)*nbData+1:s*nbData) = (real(V*D^.5) * randn(nbData,1))'; 
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1500,750]); 

% Plot prior samples
p1 = subplot(1,3,1); hold on; title('Original data');
for s=1:nbSamples
	plot(DataIn, Data(2,(s-1)*nbData+1:s*nbData), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end

p2 = subplot(1,3,2); hold on; title('Data generated with log-parametrization');
for s=1:nbSamples
	plot(DataIn, Data_hat1(1,(s-1)*nbData+1:s*nbData), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end

p3 = subplot(1,3,3); hold on; title('Data generated with square-parametrization');
for s=1:nbSamples
	plot(DataIn, Data_hat2(1,(s-1)*nbData+1:s*nbData), '-','lineWidth',3.5,'color',[.9 .9 .9]*rand(1));
end

linkaxes([p1,p2,p3]);

pause;
close all;
end


function [f,g] = LogLikelihood_logparam(Data, gamma, nbData, nbSamples)
f = 0;
g = zeros(3,1);

r2 = pdist2(Data(1,:)',Data(1,:)').^2;
for s = 1:nbSamples
	srg(s,:) = [(s-1)*nbData+1:s*nbData];
end

for s = 1:nbSamples
	kernel_s = exp(gamma(2,1)) .* exp(-r2(srg(s,:),srg(s,:)).*exp(gamma(1,1))) + exp(gamma(3,1)).*eye(nbData);
	prob = Data(2,srg(s,:))/kernel_s * Data(2,srg(s,:))';
	f = f + 0.5*prob + 0.5*(nbData)*log(2*pi) + 0.5* abs(det(kernel_s));

	pinvK = pinv(kernel_s);
	Kdivy = kernel_s\Data(2,srg(s,:))';
	
	dK1 = - exp(gamma(2,1)) .* exp(-r2(srg(s,:),srg(s,:)).*exp(gamma(1,1))) * 2 * exp(gamma(1,1)) .* r2(srg(s,:),srg(s,:));
	dK2 = exp(gamma(2,1)).* exp(-r2(srg(s,:),srg(s,:)).*exp(gamma(1,1)));
	dK3 = exp(gamma(3,1)).* eye(nbData);

	g(1,1) = g(1,1) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK1);
	g(2,1) = g(2,1) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK2);
	g(3,1) = g(3,1) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK3);

end
end


function [f,g] = LogLikelihood(Data, gamma, nbData, nbSamples)
f = 0;
g = zeros(3,1);

r2 = pdist2(Data(1,:)',Data(1,:)').^2;
for s = 1:nbSamples
	srg(s,:) = [(s-1)*nbData+1:s*nbData];
end

for s = 1:nbSamples
		kernel_s = gamma(2).*gamma(2) .* exp(-r2(srg(s,:),srg(s,:)).*gamma(1).*gamma(1)) + gamma(3).*gamma(3).*eye(nbData);
		prob = Data(2,srg(s,:))/kernel_s * Data(2,srg(s,:))';
		f = f + 0.5*prob + 0.5*(nbData)*log(2*pi) + 0.5* abs(det(kernel_s));
	
		pinvK = pinv(kernel_s);
		Kdivy = kernel_s\Data(2,srg(s,:))';
		
		dK1 = - gamma(2).*gamma(2) .* exp(-r2(srg(s,:),srg(s,:)).*gamma(1).*gamma(1)) * 2 * gamma(1) .* r2(srg(s,:),srg(s,:));
		dK2 = 2 .* gamma(2).* exp(-r2(srg(s,:),srg(s,:)).*gamma(1).*gamma(1));
		dK3 = 2 .* gamma(3).* eye(nbData);
		
		g(1,:) = g(1,:) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK1);
		g(2,:) = g(2,:) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK2);
		g(3,:) = g(3,:) - 0.5*trace((Kdivy*Kdivy'-pinvK)*dK3);
	end
end