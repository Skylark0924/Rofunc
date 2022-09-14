function model = IPRA_GMM(Data, model, dispFlag)
% Estimation of Gaussian mixture model (GMM) parameters with iterative pairwise replacement algorithm (IPRA)
% Sylvain Calinon, 2017

if nargin<3
	dispFlag=0;
end

[nbVar,nbData] = size(Data);
nbStates = model.nbStates;


%% GMM parameters estimation with IPRA (version with full covariances) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = nbData;
model.Priors = ones(nbData,1) / nbData;
model.Mu = Data;
rho = trace(cov(Data))^.5 / (nbVar*nbData/5);
model.Sigma = repmat(eye(nbVar)*rho, [1,1,nbData]); 

if dispFlag==1
	figure('position',[10,10,1200,1200]); 
end

tic
while model.nbStates > nbStates
	%Compute x_i - x_j for all i,j 
	Emat = kron(ones(model.nbStates,1),model.Mu) - kron(ones(1,model.nbStates),model.Mu(:));
	E = reshape(Emat, nbVar, model.nbStates^2);
	%Keep only non-redundant entries in the distance error matrix, by first computing the indices of non-redundant entries in the distance error matrix
	M = tril(ones(model.nbStates),-1);
	idm = find(M(:));
	[ix,iy] = ind2sub([model.nbStates,model.nbStates],idm);
	E = E(:,idm);
	
% 	%% Compute minimal distances expressed as (x_i-x_j)' * S^-1 * (x_i-x_j), with S = S_i + S_j (Bhattacharyya distance)
% 	%%%%%%%%%%%
% 	
% 	%Version 1 (fastest)
% 	S = model.Sigma(:,:,ix) + model.Sigma(:,:,iy);
% 	Q = kron(speye(length(ix)), ones(model.nbVar));
% 	Q(logical(Q)) = S(:);
% 	E = reshape(Q\E(:), model.nbVar, length(ix));
% 
% % 	%Version 2
% % 	[ix,iy] = ind2sub([model.nbStates,model.nbStates],idm);
% % 	for i=1:length(ix)
% % 		[V,D] = eig(model.invSigma(:,:,ix(i)) + model.invSigma(:,:,iy(i)));
% % 		E(:,i) = (V*D^.5*V') * E(:,i);
% % 	end
% 
% % 	%Version 3
% % 	[ix,iy] = ind2sub([model.nbStates,model.nbStates],idm);
% % 	S = model.invSigma(:,:,ix) + model.invSigma(:,:,iy);
% % 	Q = (kron(ones(length(ix),1), reshape(S, model.nbVar, model.nbVar*length(ix)))) .* kron(eye(length(ix)), ones(model.nbVar));
% % 	E = reshape(Q*E(:), model.nbVar, length(ix));
% 
% 	[~,im] = min(sum(E.*E,1)); 
% 	[id(1),id(2)] = ind2sub([model.nbStates,model.nbStates],idm(im));
	
	%% Compute minimal distances expressed as Wasserstein distance between two Gaussians: https://en.wikipedia.org/wiki/Wasserstein_metric 
	%% d = (x_i-x_j)' * (x_i-x_j) + trace(S_i + S_j - 2 .* (S_j^.5 * S_i * S_j^.5)^.5), there the tr() part is called the Bures metric 
	%% (see e.g. p.34 of computational optimal control book), corresponding to the Hellinger distance when the matrices are diagonal
	%%%%%%%%%%%
	dcov = zeros(1,length(ix));
	for i=1:length(ix)
		S1 = model.Sigma(:,:,ix(i)); 
		S2 = model.Sigma(:,:,iy(i));
		dcov(i) = trace(S1 + S2 - 2 .* (S2^.5 * S1 * S2^.5)^.5);
	end
	[~,im] = min(sum(E.*E,1) + dcov); 
	[id(1),id(2)] = ind2sub([model.nbStates,model.nbStates],idm(im));
	
	if dispFlag==1 && model.nbStates < 50
		clf; hold on; axis off;
		for i=1:model.nbStates
			plotGMM(model.Mu(:,i), model.Sigma(:,:,i), [0 .8 0],.2);
		end
		for i=1:2
			plotGMM(model.Mu(:,id(i)), model.Sigma(:,:,id(i)), [.8 0 0],.2);
		end
		plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.75 .75 .75]);
		axis equal;
		drawnow;
	end
	
	%Merge the two closest Gaussians (law of total covariance), and put the merged component in id(1)
	w = model.Priors(id);
	w = w / sum(w);
	model.Sigma(:,:,id(1)) = w(1) * model.Sigma(:,:,id(1)) + w(2) * model.Sigma(:,:,id(2)) + w(1) * w(2) * (model.Mu(:,id(1))-model.Mu(:,id(2))) * (model.Mu(:,id(1))-model.Mu(:,id(2)))';
% 	%Alternative computation (providing the same result)
% 	MuTmp = w(1) * model.Mu(:,id(1)) + w(2) * model.Mu(:,id(2));
% 	model.Sigma(:,:,id(1)) = w(1) * model.Sigma(:,:,id(1)) + w(2) * model.Sigma(:,:,id(2)) + w(1) * (model.Mu(:,id(1)) * model.Mu(:,id(1))') + w(2) * (model.Mu(:,id(2)) * model.Mu(:,id(2))') - (MuTmp * MuTmp');
	model.Mu(:,id(1)) = model.Mu(:,id) * w;
	model.Priors(id(1)) = sum(model.Priors(id));
	
	%Move the last component in id(2)
	model.Priors(id(2)) = model.Priors(end);
	model.Mu(:,id(2)) = model.Mu(:,end);
	model.Sigma(:,:,id(2)) = model.Sigma(:,:,end);
	
	%Remove the last component
	model.Priors = model.Priors(1:end-1);
	model.Mu = model.Mu(:,1:end-1);
	model.Sigma = model.Sigma(:,:,1:end-1);  
	
	%Decrease number of states
	model.nbStates = model.nbStates - 1;
	
end %while
toc

%Normalizing of Priors
model.Priors = model.Priors / sum(model.Priors);

end


% %% GMM parameters estimation with IPRA and Hellinger distances (naive slow implementation with full covariances) 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Initialization of the model (as kernel method)
% model.nbStates = nbData;
% rho = trace(cov(Data))^.5 / (model.nbVar*nbData/5);
% model.Mu = Data;
% model.Sigma = repmat(eye(model.nbVar)*rho, [1 1 nbData]);
% model.detSigma = ones(nbData,1) * rho^2;
% model.Priors = ones(nbData,1)/nbData;
% model.C = (2*(2*pi)^.5)^model.nbVar;
% 
% if dispFlag==1
% 	figure; hold on; axis off;
% 	plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.75 .75 .75]);
% 	hf = plotGMM(model.Mu, model.Sigma, [0 .8 0]);
% 	axis equal;
% 	drawnow;
% end
% 	
% while model.nbStates > nbStates
% 	%Find the two closest Gaussians (naive implementation)
% 	minH = intmax;
% 	tic
% 	for i=1:model.nbStates
% 		for j=i+1:model.nbStates
% 			H = HellingerDist(model,i,j);
% 			if H<minH
% 				id = [i,j];
% 				minH = H;
% 			end
% 		end
% 	end
% 	toc
% 	%Merge the two closest Gaussians
% 	w1 = model.Priors(id(1)) / (model.Priors(id(1))+model.Priors(id(2)));
% 	w2 = model.Priors(id(2)) / (model.Priors(id(1))+model.Priors(id(2)));
% 	%Put the merge component in id(1)
% 	model.Sigma(:,:,id(1)) = w1 * model.Sigma(:,:,id(1)) + w2 * model.Sigma(:,:,id(2)) + ...
% 		w1 * w2 * (model.Mu(:,id(1))-model.Mu(:,id(2))) * (model.Mu(:,id(1))-model.Mu(:,id(2)))';
% 	model.Mu(:,id(1)) = w1 * model.Mu(:,id(1)) + w2 * model.Mu(:,id(2));
% 	model.Priors(id(1)) = model.Priors(id(1)) + model.Priors(id(2));
% 	model.detSigma(id(1)) = det(model.Sigma(:,:,id(1)));
% 	%Move the last component in id(2)
% 	model.Priors(id(2)) = model.Priors(end);
% 	model.Mu(:,id(2)) = model.Mu(:,end);
% 	model.Sigma(:,:,id(2)) = model.Sigma(:,:,end);
% 	model.detSigma(id(2)) = model.detSigma(end);
% 	%Clean the last component
% 	model.Priors(end) = [];
% 	model.Mu(:,end) = [];
% 	model.Sigma(:,:,end) = [];
% 	model.detSigma(end) = [];
% 	%Decrease number of states
% 	model.nbStates = model.nbStates-1;
% 	
% 	if dispFlag==1 %&& model.nbStates<14
% 		delete(hf);
% 		hf = plotGMM(model.Mu, model.Sigma, [0 .8 0]);
% 		%for i=1:model.nbStates
% 		%	hf = [hf text(model.Mu(1,i),model.Mu(2,i),num2str(model.Priors(i)))];
% 		%end
% 		drawnow;
% 	end
% end %while
%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function d = HellingerDist(model,i,j)
% 	%see e.g. https://en.wikipedia.org/wiki/Hellinger_distance
% 	d = model.Priors(i) + model.Priors(j) - 2 * (model.Priors(i)*model.Priors(j))^.5 * model.C * ...
% 		model.detSigma(i)^.25 * model.detSigma(j)^.25 * gaussPDF(model.Mu(:,i), model.Mu(:,j), 2*model.Sigma(:,:,i)+2*model.Sigma(:,:,j));
% end




