function [bmx, ALPHA, S, alpha] = hsmm_fwd_step_hsum(Data, m, bmx, ALPHA, S,varargin)
%Forward variable iteration
%Sylvain Calinon, 2014 

% Check if we need to evaluate marginal probability:
if nargin ==5
	%  Input for evaluation of GMM not specified, use all variables
	in = 1:size(m.Mu,1);
else
	in =varargin{1};
end

nbD = size(m.Pd, 2);
nbStates = size(m.Sigma, 3);

for i=1:nbStates
  Btmp(i,1) = gaussPDF(Data(in), m.Mu(in,i), m.Sigma(in,in,i))+1e-12; 
end
Btmp = Btmp/sum(Btmp);

%Fast computation
ALPHA = [repmat(S(:,end), 1, nbD-1) .* m.Pd(:,1:nbD-1) + repmat(bmx(:,end), 1, nbD-1) .* ALPHA(:,2:nbD), ...
  S(:,end) .* m.Pd(:,nbD)];	%Equation (12)

% %Slow computation (but easier to read)
% for i=1:nbD-1
%   ALPHA(:,i) = S(:,end) .* m.Pd(:,i) + bmx(:,end) .* ALPHA(:,i+1);	%Equation (12)
% end
% ALPHA(:,nbD) = S(:,end) .* m.Pd(:,nbD);
% 
% ALPHA_SUM = zeros(nbStates,1);
% for i=1:nbStates
%   ALPHA_SUM(i) = sum(ALPHA(i,:));
% end

r = Btmp' * sum(ALPHA,2);     %Equation (3)
bmx = [bmx, Btmp ./ r];	      %Equation (2)
E = bmx(:,end) .* ALPHA(:,1); %Equation (5)
S = [S,m.Trans' * E];         %Equation (6)

alpha = Btmp .* sum(ALPHA,2); %Forward variable
alpha = alpha / sum(alpha);
end
