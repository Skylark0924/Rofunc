function [bmx, ALPHA, S, alpha] = hsmm_fwd_init_hsum(Data, m,varargin)
%Forward variable initialization
%Sylvain Calinon, 2014 


% Check if we need to evaluate marginal probability:
if nargin ==2
	in = 1:size(m.Mu,1);
else
	in = varargin{1};
end

for i=1:size(m.Sigma,3)
  Btmp(i,1) = gaussPDF(Data(in,:), m.Mu(in,i), m.Sigma(in,in,i)) +1e-12;
end
Btmp = Btmp / sum(Btmp);

ALPHA = repmat(m.StatesPriors, 1, size(m.Pd,2)) .* m.Pd;	% Equation (13)
r = Btmp' * sum(ALPHA,2);    % Equation (3) 
bmx(:,1) = Btmp ./ r;        % Equation (2)
E = bmx .* ALPHA(:,1);       % Equation (5)
S = m.Trans' * E;            % Equation (6)

alpha = Btmp .* sum(ALPHA,2);% Forward variable

end