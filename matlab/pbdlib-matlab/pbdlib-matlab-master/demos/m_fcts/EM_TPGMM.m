function model = EM_TPGMM(Data, model)
% Training of a task-parameterized Gaussian mixture model (GMM) with an expectation-maximization (EM) algorithm.
% The approach allows the modulation of the centers and covariance matrices of the Gaussians with respect to
% external parameters represented in the form of candidate coordinate systems.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
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
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-5; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,3);

diagRegularizationFactor = 1E-5; %Optional regularization term
%diagRegularizationFactor = 0; %Optional regularization term

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	
	%M-step
	for i=1:model.nbStates
		
		%Update Priors
		model.Priors(i) = sum(sum(GAMMA(i,:))) / nbData;
		
		for m=1:model.nbFrames
			%Matricization/flattening of tensor
			DataMat=[];
			DataMat(1:model.nbVars(m),:) = Data(1:model.nbVars(m),m,:);
			
			%Update Mu
			model.Mu(1:model.nbVars(m),m,i) = DataMat * GAMMA2(i,:)';
			
			%Update Sigma (regularization term is optional)
			DataTmp = DataMat - repmat(model.Mu(1:model.nbVars(m),m,i),1,nbData);
			model.Sigma(1:model.nbVars(m),1:model.nbVars(m),m,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVars(m))*diagRegularizationFactor;
		end
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / size(L,2);
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lik, GAMMA, GAMMA0] = computeGamma(Data, model)
nbData = size(Data, 3);
Lik = ones(model.nbStates, nbData);
GAMMA0 = zeros(model.nbStates, model.nbFrames, nbData);
for i=1:model.nbStates
	for m=1:model.nbFrames
		DataMat=[];
		DataMat(:,:) = Data(1:model.nbVars(m),m,:); %Matricization/flattening of tensor
		GAMMA0(i,m,:) = gaussPDF(DataMat, model.Mu(1:model.nbVars(m),m,i), model.Sigma(1:model.nbVars(m),1:model.nbVars(m),m,i));
		Lik(i,:) = Lik(i,:) .* squeeze(GAMMA0(i,m,:))';
	end
	Lik(i,:) = Lik(i,:) * model.Priors(i);
end
GAMMA = Lik ./ repmat(sum(Lik,1)+realmin, size(Lik,1), 1);
end
