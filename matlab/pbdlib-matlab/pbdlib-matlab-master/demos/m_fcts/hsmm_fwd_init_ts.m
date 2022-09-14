function [ALPHA, S, alpha] = hsmm_fwd_init_ts(model)
%Forward variable by using only temporal/sequential information from Trans and Pd
%Sylvain Calinon, 2014 

ALPHA = repmat(model.StatesPriors, 1, size(model.Pd,2)) .* model.Pd; %Equation (13)
S = model.Trans' * ALPHA(:,1);	%Equation (6)
alpha = sum(ALPHA,2); %Forward variable
