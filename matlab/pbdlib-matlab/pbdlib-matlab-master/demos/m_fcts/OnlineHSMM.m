function [model] = OnlineHSMM(model, demo, plot_pred)
% Note that this needs to be called only after OnlineEMDP
if nargin < 3
    plot_pred = 0;
end
nVar = size(model.Mu, 1);
nStates = size(model.Sigma, 3);
if ~isfield(model,'Trans') % need to init the model
    disp('init model');
    model.Trans = zeros(nStates, nStates);
    model.hsmm_Trans_c = zeros(nStates, nStates);
    model.StatesPriors = zeros(nStates, 1);
    model.hsmm_StatesPriors_c = zeros(nStates, 1);
    model.Mu_Pd = zeros(1, nStates);
    model.Sigma_Pd = zeros(1, 1, nStates);
    model.running_c = zeros(nStates, 1);
    model.running_M2 = zeros(nStates, 1);
elseif size(model.Trans) ~= nStates % neet to resize the model
    disp('resize model');
    model.Trans(nStates, nStates) = 0;
    model.hsmm_Trans_c(nStates, nStates) = 0;
    model.StatesPriors(1, nStates) = 0;
    model.hsmm_StatesPriors_c(1, nStates) = 0;
    model.Mu_Pd(1, nStates) = 0;
    model.Sigma_Pd(1, 1, nStates) = 0;
    model.running_c(nStates, 1) = 0;
    model.running_M2(nStates, 1) = 0;
end
nPoints = size(demo,2);
if nPoints < 10
    disp('Demo too short!');
    return;
end
h = zeros(nStates,nPoints);
for i=1:nStates
    h(i,:)=gaussPDF(demo, model.Mu(:,i), model.Sigma(:,:,i));
end
[~,state_seq] = max(h);
% compute state priors
model.hsmm_StatesPriors_c(state_seq(1)) = model.hsmm_StatesPriors_c(state_seq(1)) + 1;
model.StatesPriors = model.hsmm_StatesPriors_c / sum(model.hsmm_StatesPriors_c);
% compute duration probabilities
currState = state_seq(1);
cnt = 0;
for t = 1:length(state_seq)
    if state_seq(t) == currState && t~=length(state_seq)
        cnt = cnt + 1; % counter is the duration really
    else % update statistics
        if t==length(state_seq)
            cnt = cnt + 1; % reached the end
        end
        model.running_c(currState) = model.running_c(currState) + 1;
        delta = cnt - model.Mu_Pd(1,currState);
        model.Mu_Pd(1,currState) = model.Mu_Pd(1,currState) + (delta / model.running_c(currState));
        model.running_M2(currState) =  model.running_M2(currState) + delta * (cnt - model.Mu_Pd(1,currState));
        if model.running_c(currState) < 2
            model.Sigma_Pd(1,1,currState) = 1;
        else
            model.Sigma_Pd(1,1,currState) =  model.running_M2(currState) / (model.running_c(currState) - 1);
        end
        % record the transition
        model.hsmm_Trans_c(currState, state_seq(t)) = model.hsmm_Trans_c(currState, state_seq(t)) + 1;
        cnt = 1; % moving on
        currState = state_seq(t);
    end
end
for i= 1 : nStates % Transition probabilities
    row_sum = sum(model.hsmm_Trans_c(i,:));
    if row_sum > 1
        model.Trans(i,:) = model.hsmm_Trans_c(i,:) / row_sum;
    else
        model.Trans(i,:) = model.hsmm_Trans_c(i,:);
    end
end
if (plot_pred)
    pred = sample_hsmm_lqr(model, state_seq(1), nPoints);
    plot(pred(1,:),pred(2,:),'-b','LineWidth',1);
end
end

