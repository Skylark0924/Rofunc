function r = estimateAttractorPathTPGMR(DataIn, model, r)
% Estimation of attractor path with TP-GMR.
% João Silvério, 2016
%
% In TP-GMR:
%   - GMR is computed in each of the local frames
%   - GMR outputs one Gaussian distribution per frame, which is linearly
%   transformed according to the TPs, followed by Gaussian product
%
% As opposed to TP-GMM, where:
%   - The product of linearly transformed GMMs is performed
%   - GMR is performed in the resulting GMM

in = 1:size(DataIn,1);

%% TP-GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The first step is to compute GMR in all frames
% (some parameters of the local models are the same for all frames:)

% initialize local model structure
LocalModel = model; % all the parameters are the same except Mu and Sigma (defined in the for loop)

% create a TP-GMM with one state to compute Gaussian product (comply with productTPGMM0.m)
OneStateModel = model;
OneStateModel.nbStates  = 1;
OneStateModel.nbVar     = model.nbVar;
OneStateModel.Priors	= 1;
OneStateModel.Mu = [];
OneStateModel.Sigma = [];
OneStateModel.nbVars = OneStateModel.nbVars-1;

% go through all frames
for n=1:model.nbFrames
    out = in(end)+1:model.nbVars(n);
    % project input data on the frames - key to TP-GMR
    DataInLocal = r.p(n).A(in,in)\(DataIn-r.p(n).b(in));    % affine transform
    
    % get the centers and covariances of the GMM in the n-th frame
    LocalModel.Mu		= squeeze(model.Mu(:,n,:));
	LocalModel.Sigma	= squeeze(model.Sigma(:,:,n,:));

    % compute GMR for the n-th GMM
    [OneStateModel.Mu(:,n,:), OneStateModel.Sigma(:,:,n,:)] = GMR(LocalModel, DataInLocal, in, out);

    % redefine the task parameters to contain only the output dimensions
    % (relevant for the new Gaussian product)
    r.p(n).A = r.p(n).A(2:model.nbVarOut+1,in(end)+1:end);
    r.p(n).b = r.p(n).b(2:model.nbVarOut+1);
%     plotGMM(squeeze(OneStateModel.Mu([1,2],n,:)), squeeze(OneStateModel.Sigma([1,2],[1,2],n,:)), [0 .7*n/model.nbFrames 0]);
%     pause
end

%         DataInLocal
% The second step is to compute the product of the resulting lin. transf. Gaussians
% [r.currTar, r.currSigma] = productTPGMM0(OneStateModel, r.p); 
[r.currTar, r.currSigma, r.localModel] = productTPGMR0(OneStateModel, r.p); 
r.nbStates = 1;
r.Priors   = 1;

% plotGMM(r.currTar([1,2],:), r.currSigma([1,2],[1,2],:), [0 0 0.7]);
% pause
