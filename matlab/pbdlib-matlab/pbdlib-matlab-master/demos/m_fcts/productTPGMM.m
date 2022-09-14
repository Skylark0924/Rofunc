function [Mu, Sigma] = productTPGMM(model, p)
% Compute the product of Gaussians for a task-parametrized model where the
% set of parameters are stored in the variable 'p'.
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


%diagRegularizationFactor = 1E-6; %Optional regularization term
diagRegularizationFactor = 1E-4; %Optional regularization term

for i=1:model.nbStates
	%Reallocating
	SigmaTmp = zeros(model.nbVar);
	MuTmp = zeros(model.nbVar,1);
	%Product of Gaussians
	for m=1:model.nbFrames
		MuP = p(m).A * model.Mu(1:model.nbVars(m),m,i) + p(m).b;
		SigmaP = p(m).A * model.Sigma(1:model.nbVars(m),1:model.nbVars(m),m,i) * p(m).A' + eye(model.nbVar)*diagRegularizationFactor;
		SigmaTmp = SigmaTmp + inv(SigmaP);
		MuTmp = MuTmp + SigmaP\MuP;
	end
	Sigma(:,:,i) = inv(SigmaTmp);
	Mu(:,i) = Sigma(:,:,i) * MuTmp;
end
