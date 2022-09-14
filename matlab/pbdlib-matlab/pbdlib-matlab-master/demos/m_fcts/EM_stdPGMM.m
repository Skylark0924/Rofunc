function [model, s, LL] = EM_stdPGMM(s, model)
% Training of a parametric Gaussian mixture model (PGMM) with expectation-maximization (EM).
% The approach is inspired by Wilson and Bobick (1999), with an implementation applied to 
% the special case of Gaussian mixture models (GMM).
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
% @article{Wilson99,
%   author="Wilson, A. D. and Bobick, A. F.",
%   title="Parametric Hidden {M}arkov Models for Gesture Recognition",
%   journal="{IEEE} Trans. on Pattern Analysis and Machine Intelligence",
%   year="1999",
%   volume="21",
%   number="9",
%   pages="884--900"
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
nbMinSteps = 10; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-50; %Likelihood increase threshold to stop the algorithm

diagRegularizationFactor = 1E-4; %Optional regularization term for the covariance update

%Initialization of the parameters
nbSamples = length(s);
nbData=0;
for n=1:nbSamples
	nbData = nbData + s(n).nbData;
end
nbVarParams = size(s(1).OmegaMu,1);

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-STEP
	[s, GAMMA] = computeGamma(s, model); %See 'computeGamma' function below
	
	%M-STEP
	for i=1:model.nbStates
		
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:))/nbData;
		
		%Update Zmu
		model.ZMu(:,:,i) = zeros(model.nbVar,nbVarParams);
		sumTmp = zeros(nbVarParams,nbVarParams);
		for n=1:nbSamples
			model.ZMu(:,:,i) = model.ZMu(:,:,i) + (s(n).Data * diag(s(n).GAMMA(i,:)) * repmat(s(n).OmegaMu',s(n).nbData,1));
			sumTmp = sumTmp + (s(n).OmegaMu*s(n).OmegaMu') * (sum(s(n).GAMMA(i,:)));
		end
		model.ZMu(:,:,i) = model.ZMu(:,:,i) * pinv(sumTmp + eye(nbVarParams)*diagRegularizationFactor); 
		
		%Update Sigma
		model.Sigma(:,:,i) = zeros(model.nbVar);
		sumTmp = 0;
		for n=1:nbSamples
			MuTmp = model.ZMu(:,:,i) * s(n).OmegaMu;
			Data_tmp = s(n).Data - repmat(MuTmp,1,s(n).nbData);
			model.Sigma(:,:,i) = model.Sigma(:,:,i) + Data_tmp * diag(s(n).GAMMA(i,:)) * Data_tmp';
			sumTmp = sumTmp + sum(s(n).GAMMA(i,:));
		end
		model.Sigma(:,:,i) = (model.Sigma(:,:,i) + eye(model.nbVar)*diagRegularizationFactor) / sumTmp;
		
	end
	
	%Computes the average log-likelihood through the ALPHA scaling factors
	LL(nbIter)=0; sz=0;
	for n=1:nbSamples
		LL(nbIter) = LL(nbIter) + sum(log(sum(s(n).GAMMA0,1)));
		sz = sz + s(n).nbData;
	end
	LL(nbIter) = LL(nbIter) / sz;
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
function [s, GAMMA] = computeGamma(s, model)
nbSamples = length(s);
nbStates = size(model.Sigma,3);
%Observation probabilities
GAMMA0=[];
for n=1:nbSamples
	for i=1:nbStates
		MuTmp = model.ZMu(:,:,i) * s(n).OmegaMu;
		s(n).GAMMA0(i,:) = model.Priors(i) * gaussPDF(s(n).Data,MuTmp,model.Sigma(:,:,i));
	end
	GAMMA0 = [GAMMA0 s(n).GAMMA0];
end
%Normalization
GAMMA = GAMMA0 ./ repmat(sum(GAMMA0,1)+realmin,size(GAMMA0,1),1);
%Data reshape
nTmp=1;
for n=1:nbSamples
	s(n).GAMMA = GAMMA(:,[nTmp:nTmp+size(s(n).GAMMA0,2)-1]);
	nTmp = nTmp+size(s(n).GAMMA,2);
end
end
