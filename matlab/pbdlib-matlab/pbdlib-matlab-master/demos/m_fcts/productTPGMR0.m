function [Mu, Sigma, localModel] = productTPGMR0(model, p)
% Compute the product of Gaussians for a TP-GMR (nbStates = 1)
%
% Copied from productTPGMM0.m but returns the linearly transformed models.
%
% João Silvério, 2016

diagRegularizationFactor = 1.0E-4;
colorPlot(1,:) = [0.7 0 0];
colorPlot(2,:) = [0 0.7 0];
% hold on

for i=1:model.nbStates
	% Reallocating
	SigmaTmp = zeros(model.nbVarOut);
	MuTmp = zeros(model.nbVarOut,1);
	% Product of Gaussians
	for m=1:model.nbFrames
		MuP = p(m).A * model.Mu(1:model.nbVars(m),m,i) + p(m).b;
        SigmaP = p(m).A * model.Sigma(1:model.nbVars(m),1:model.nbVars(m),m,i) * p(m).A' + eye(model.nbVarOut)*diagRegularizationFactor;
		SigmaP = diag(diag(SigmaP));

        SigmaTmp = SigmaTmp + inv(SigmaP);
		MuTmp = MuTmp + SigmaP\MuP;
		
		% Stores the linearly transformed estimates from each frame
		localModel(m).Mu = MuP;
		localModel(m).Sigma = SigmaP;
	end
	Sigma(:,:,i) = inv(SigmaTmp);
	Mu(:,i) = Sigma(:,:,i) * MuTmp;
end

% plotGMM(Mu([1,2],:), Sigma([1,2],[1,2],:), [0 0 0.7]);
% pause
% clf