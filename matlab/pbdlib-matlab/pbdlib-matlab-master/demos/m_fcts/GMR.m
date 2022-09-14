function [expData, expSigma, H] = GMR(model, DataIn, in, out)
% Gaussian mixture regression (GMR)
%
% If this code is useful for your research, please cite the related publication:
%
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
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


nbData = size(DataIn,2);
nbVarOut = length(out);

if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-8; %Regularization term is optional
end

MuTmp = zeros(nbVarOut, model.nbStates);
expData = zeros(nbVarOut, nbData);
expSigma = zeros(nbVarOut, nbVarOut, nbData);
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) .* gaussPDF(DataIn(:,t), model.Mu(in,i), model.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) ./ sum(H(:,t)+realmin);
	%Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i) / model.Sigma(in,in,i) * (DataIn(:,t)-model.Mu(in,i));
		expData(:,t) = expData(:,t) + H(i,t) .* MuTmp(:,i);
	end
	%Compute conditional covariances
	for i=1:model.nbStates
		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i) / model.Sigma(in,in,i) * model.Sigma(in,out,i);
		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) .* (SigmaTmp + MuTmp(:,i) * MuTmp(:,i)');
	end
	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t) * expData(:,t)' + eye(nbVarOut) * model.params_diagRegFact; 
end

