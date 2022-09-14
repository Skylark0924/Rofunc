function [ALPHA, S, alpha] = hsmm_fwd_step_ts(model, ALPHA, S)
%Forward variable by using only temporal/sequential information from Trans and Pd
%Sylvain Calinon, 2014

t = size(ALPHA,2) + 1;
nbD = size(model.Pd, 2);

%Fast computation
ALPHA = [repmat(S(:,end), 1, nbD-1) .* model.Pd(:,1:nbD-1) + ALPHA(:,2:nbD), ...
	S(:,end) .* model.Pd(:,nbD)]; %Equation (12)

% %Slow computation (but more readable)
% for i=1:nbD-1
%   ALPHA(:,i) = S(:,end) .* model.Pd(:,i) + ALPHA(:,i+1);	%Equation (12)
% end
% ALPHA(:,nbD) = S(:,end) .* model.Pd(:,nbD);

S = [S, model.Trans' * ALPHA(:,1)];	%Equation (6)
alpha = sum(ALPHA, 2); %Forward variable

