function [pred] = sample_hsmm_lqr(model, start_state, traj_length, start_point)
if nargin<4
    start_point = model.Mu(:,start_state);
end
nStates = size(model.Sigma, 3);
model.nbStates = nStates;
nbData = traj_length; %
nbD = round(2 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)
%Precomputation of duration probabilities
for i=1:nStates
    model.Pd(i,:) = gaussPDF([1:nbD], model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i));
    %The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
    model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
end

%Manual reconstruction of sequence for HSMM based on stochastic sampling
nbSt=0; currTime=0; iList=[];
h = zeros(model.nbStates,nbData);
while currTime<nbData
    nbSt = nbSt+1;
    if nbSt==1
        %[~,iList(1)] = max(model.StatesPriors.*rand(model.nbStates,1));
        iList(1) = start_state;
        h1 = ones(1,nbData);
    else
        h1 = [zeros(1,currTime), cumsum(model.Pd(iList(end-1),:)), ones(1,max(nbData-currTime-nbD,0))];
        currTime = currTime + round(model.Mu_Pd(1,iList(end-1)));
    end
    h2 = [ones(1,currTime), 1-cumsum(model.Pd(iList(end),:)), zeros(1,max(nbData-currTime-nbD,0))];
    h(iList(end),:) = h(iList(end),:) + min([h1(1:nbData); h2(1:nbData)]);
    [~,iList(end+1)] = max(model.Trans(iList(end),:).*rand(1,model.nbStates));
end
h = h ./ repmat(sum(h,1),model.nbStates,1);

[~,qList] = max(h);

% Iterative LQR reproduction (finite horizon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 1; %Number of static & dynamic features (D=1 for [x1,x2])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.rfactor = 1E-2;	%Control cost in LQR
model.dt = 0.01; %Time step duration

%Dynamical System settings (discrete version)
A = kron([1, model.dt; 0, 1], eye(model.nbVarPos));
B = kron([0; model.dt], eye(model.nbVarPos));
%C = kron([1, 0], eye(model.nbVarPos));
%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;

P = zeros(model.nbVarPos*2,model.nbVarPos*2,nbData);
P(1:model.nbVarPos,1:model.nbVarPos,end) = inv(model.Sigma(:,:,qList(nbData)));
d = zeros(model.nbVarPos*2, nbData);
Q = zeros(model.nbVarPos*2);
for t=nbData-1:-1:1
    Q(1:model.nbVarPos,1:model.nbVarPos) = inv(model.Sigma(:,:,qList(t)));
    P(:,:,t) = Q - A' * (P(:,:,t+1) * B / (B' * P(:,:,t+1) * B + R) * B' * P(:,:,t+1) - P(:,:,t+1)) * A;
    d(:,t) = (A' - A'*P(:,:,t+1) * B / (R + B' * P(:,:,t+1) * B) * B' ) * (P(:,:,t+1) * ...
        ( A * [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - [model.Mu(:,qList(t+1)); zeros(model.nbVarPos,1)] ) + d(:,t+1));
end
%Reproduction with feedback (FB) and feedforward (FF) terms
X = [start_point; zeros(model.nbVarPos,1)]; %[model.Mu(:,qList(1)); zeros(model.nbVarPos,1)];
r.X0 = X;
for t=1:nbData
    r.Data(:,t) = X; %Log data
    K = (B' * P(:,:,t) * B + R) \ B' * P(:,:,t) * A; %FB term
    M = -(B' * P(:,:,t) * B + R) \ B' * (P(:,:,t) * ...
        (A * [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)]) + d(:,t)); %FF term
    u = K * ([model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - X) + M; %Acceleration command with FB and FF terms
    X = A * X + B * u; %Update of state vector
end

pred = r.Data;
% plot(r(n).Data(1,:),r(n).Data(2,:),'-b','LineWidth',2);
end