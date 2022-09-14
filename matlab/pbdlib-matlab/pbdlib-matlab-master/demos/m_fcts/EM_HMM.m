function [model, GAMMA2, LL] = EM_HMM(s, model)
% Estimation of HMM parameters with an EM algorithm.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Rozo16Frontiers,
%   author="Rozo, L. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
%   title="Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration",
%   journal="Frontiers in Robotics and {AI}",
%   year="2016",
%   month="June",
%   volume="3",
%   number="30",
%   pages="1--11",
%   doi="10.3389/frobt.2016.00030"
% }
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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


%Parameters of the EM algorithm
if ~isfield(model,'params_nbMinSteps')
	model.params_nbMinSteps = 5; %Minimum number of iterations allowed
end
if ~isfield(model,'params_nbMaxSteps')
	model.params_nbMaxSteps = 50; %Maximum number of iterations allowed
end
if ~isfield(model,'params_maxDiffLL')
	model.params_maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
end
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-8; %Regularization term is optional
end
if ~isfield(model,'params_updateComp') || length(model.params_updateComp)<4
	model.params_updateComp = ones(4,1); %Mu,Sigma,Pi,A
end	


%Initialization of the parameters
nbSamples = length(s);
Data = [];
for n=1:nbSamples
	Data = [Data s(n).Data];
	s(n).nbData = size(s(n).Data,2);
end
[nbVar, nbData] = size(Data);

for nbIter=1:model.params_nbMaxSteps
	fprintf('.');
	
	%E-step
	for n=1:nbSamples
		
		%Emission probabilities
		for i=1:model.nbStates
			%s(n).B(i,:) = model.Priors(i) * gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
			s(n).B(i,:) = gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
		end
		
		%Forward variable ALPHA (rescaled)
		s(n).ALPHA(:,1) = model.StatesPriors .* s(n).B(:,1);
		%Scaling to avoid underflow issues
		s(n).c(1) = 1 / sum(s(n).ALPHA(:,1)+realmin);
		s(n).ALPHA(:,1) = s(n).ALPHA(:,1) * s(n).c(1);
		for t=2:s(n).nbData
			s(n).ALPHA(:,t) = (s(n).ALPHA(:,t-1)'*model.Trans)' .* s(n).B(:,t); 
			%Scaling to avoid underflow issues
			s(n).c(t) = 1 / sum(s(n).ALPHA(:,t)+realmin);
			s(n).ALPHA(:,t) = s(n).ALPHA(:,t) * s(n).c(t);
		end
		
		%Backward variable BETA (rescaled)
		s(n).BETA(:,s(n).nbData) = ones(model.nbStates,1) * s(n).c(end); %Rescaling
		for t=s(n).nbData-1:-1:1
			s(n).BETA(:,t) = model.Trans * (s(n).BETA(:,t+1) .* s(n).B(:,t+1));
			s(n).BETA(:,t) = min(s(n).BETA(:,t) * s(n).c(t), realmax); %Rescaling
		end
		
		%Intermediate variable GAMMA
		s(n).GAMMA = (s(n).ALPHA.*s(n).BETA) ./ repmat(sum(s(n).ALPHA.*s(n).BETA)+realmin, model.nbStates, 1); 
		
		%Intermediate variable ZETA (fast version, by considering scaling factor)
		for i=1:model.nbStates
			for j=1:model.nbStates
				s(n).ZETA(i,j,:) = model.Trans(i,j) * (s(n).ALPHA(i,1:end-1) .* s(n).B(j,2:end) .* s(n).BETA(j,2:end)); 
			end
		end
	end
	
	%Concatenation of HMM intermediary variables
	GAMMA=[]; GAMMA_TRK=[]; GAMMA_INIT=[]; ZETA=[];
	for n=1:nbSamples
		GAMMA = [GAMMA s(n).GAMMA];
		GAMMA_INIT = [GAMMA_INIT s(n).GAMMA(:,1)];
		GAMMA_TRK = [GAMMA_TRK s(n).GAMMA(:,1:end-1)];
		ZETA = cat(3,ZETA,s(n).ZETA);
	end
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, size(GAMMA,2));
	
	%M-step
	for i=1:model.nbStates
		%Update the centers
		if model.params_updateComp(1)
			model.Mu(:,i) = Data * GAMMA2(i,:)'; 
		end	
		%Update the covariance matrices
		if model.params_updateComp(2)
			Data_tmp = Data - repmat(model.Mu(:,i),1,nbData);
			model.Sigma(:,:,i) = Data_tmp * diag(GAMMA2(i,:)) * Data_tmp'; %Eq. (54) Rabiner
			%Optional regularization term
			model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(nbVar) * model.params_diagRegFact;
		end
	end
	
	%Update initial state probability vector
	if model.params_updateComp(3)
		model.StatesPriors = mean(GAMMA_INIT,2);
	end
	
	%Update transition probabilities
	if model.params_updateComp(4)
		model.Trans = sum(ZETA,3)./ repmat(sum(GAMMA_TRK,2)+realmin, 1, model.nbStates); 
	end
	
	%Compute the average log-likelihood through the ALPHA scaling factors
	LL(nbIter)=0;
	for n=1:nbSamples
		LL(nbIter) = LL(nbIter) - sum(log(s(n).c));
	end
	LL(nbIter) = LL(nbIter)/nbSamples;
	%Stop the algorithm if EM converged
	if nbIter>model.params_nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<model.params_maxDiffLL
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end

disp(['The maximum number of ' num2str(model.params_nbMaxSteps) ' EM iterations has been reached.']);



