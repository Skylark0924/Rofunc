function demo_HMM02
% Emulation of HSMM with a standard HMM 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
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
model.nbStates = 5;
nbData = 100;
nbSamples = 5;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/C.mat');
%nbSamples = length(demos);
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased([repmat(1:nbData,1,nbSamples); Data], model);
model.Mu = model.Mu(2:end,:);
model.Sigma = model.Sigma(2:end,2:end,:);

% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1-(model.nbStates/nbData);
	model.Trans(i,i+1) = model.nbStates/nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

% %Parameters refinement with EM
% model = EM_HMM(s, model);
model = EM_GMM(Data, model);

model.StatesPriors
model.Trans

%Computation of state duration probability (geometric distribution) based on transition information 
a(:,1) = model.StatesPriors;
for t=2:nbData
	a(:,t) = (a(:,t-1)'*model.Trans)'; 
end	

%Duplication of states as a way to change duration modeling
nbRep = 12;
model2 = model;
model2.nbStates = model.nbStates * nbRep;
model2.StatesPriors = kron(model.StatesPriors,[1; zeros(nbRep-1,1)]);
% model2.StatesPriors

%model2.Trans = kron(model.Trans, [1 zeros(1,nbRep-1); zeros(nbRep-1,nbRep)]);
model2.Trans = zeros(model2.nbStates);
for i=1:model2.nbStates-1
	model2.Trans(i,i) = 1-(model2.nbStates/nbData);
	model2.Trans(i,i+1) = model2.nbStates/nbData;
end

% %model2.Trans
% return
% for i=1:model2.nbStates
% 	if mod(i,nbRep)==0
% 		model2.Trans(:,i) = kron(model.Trans(:,i/nbRep), [1; zeros(nbRep-1,1)])
% 	end
% end

%Computation of state duration probability (geometric distribution) based on transition information 
a2(:,1) = model2.StatesPriors;
for t=2:nbData
	a2(:,t) = (a2(:,t-1)'*model2.Trans)'; 
end	


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,900]); 
clrmap = lines(model.nbStates);
clrmap2 = lines(model2.nbStates);

%Plot spatial data
subplot(1,3,1); hold on;
for i=1:model.nbStates
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), 1);
end
plot(Data(1,:), Data(2,:), '.', 'linewidth', 2, 'color', [.3 .3 .3]);
xlabel('x_1','fontsize',16);
ylabel('x_2','fontsize',16);
axis equal;

%Plot state duration probability
subplot(1,3,2); hold on;
for i=1:model.nbStates
	plot(a(i,:), '-','color', clrmap(i,:));
end
% for t=1:nbData
% 	plot(t, model.Trans(1,1)^(t-1)*(1-model.Trans(1,1)), 'k.'); %Geometric distribution (./ model.Trans(1,2))
% end
disp(['Average time staying in state 1: ' num2str(1/(1-model.Trans(1,1)))]);
xlabel('$t$','interpreter','latex','fontsize',16);
ylabel('$P^d_i$','interpreter','latex','fontsize',16);
%axis square;

%Plot state duration probability
subplot(1,3,3); hold on;
for i=1:model2.nbStates
	if mod(i,nbRep)~=0
		plot(a2(i,:), '-','color', min(clrmap(ceil(i/nbRep),:)+0.2,1),'linewidth',0.5);
	end
end
for i=1:model2.nbStates
	if mod(i,nbRep)==0
		plot(a2(i,:), '-','color', clrmap(i/nbRep,:),'linewidth',3);
	end
end
xlabel('$t$','interpreter','latex','fontsize',16);
ylabel('$P^d_i$','interpreter','latex','fontsize',16);
%axis square;

%print('-dpng','graphs/demo_HMM02.png');
pause;
close all;