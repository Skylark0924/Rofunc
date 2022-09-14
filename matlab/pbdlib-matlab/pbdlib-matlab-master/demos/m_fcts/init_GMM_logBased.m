function model = init_GMM_logBased(Data, model)
%Initialization based on equal intervals in ln space
%Sylvain Calinon, 2015

%Data = exp(Data);

[nbVar, nbData] = size(Data);
%diagRegularizationFactor = 1E-2;
diagRegularizationFactor = 1E-8;

TimingSep = log(linspace(exp(Data(1,1)), exp(Data(1,end)), model.nbStates+1));

for i=1:model.nbStates
	idtmp = find(Data(1,:)>=TimingSep(i) & Data(1,:)<TimingSep(i+1));
	model.Priors(i) = length(idtmp);
	model.Mu(:,i) = mean(Data(:,idtmp)');
	model.Sigma(:,:,i) = cov(Data(:,idtmp)');
	%Regularization term to avoid numerical instability
	model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(nbVar)*diagRegularizationFactor;
	
	%model.Mu(:,i) = exp(model.Mu(:,i));
	%model.Sigma(:,:,i) = exp(model.Sigma(:,:,i));

end
model.Priors = model.Priors / sum(model.Priors);

% model.Mu = exp(model.Mu + 0.5*squeeze(model.Sigma)');
% model.Sigma = reshape(squeeze(exp(model.Sigma)-1)' .* model.Mu.^2, [1 1 model.nbStates]);


