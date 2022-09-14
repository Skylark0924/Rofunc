function [yd, SigmaOut] = GPR(q, y, qd, p, covopt)
% Gaussian process regression (GPR)
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


%Kernel parameters
if nargin<4
	%p(1)=1; p(2)=1; p(3)=1E-5;
	p(1)=1; p(2)=1E-1; p(3)=1E-3;
end

%Covariance computation
if nargin<5
	covopt = 1;
end
	
diagRegularizationFactor = 1E-4; %Optional regularization term

% %Linear least-squares regression
% vOut = y * (pinv(q)*vIn);

% %Recenter data
% qmean = mean(q,2);
% q = q - repmat(qmean,1,size(q,2));
% qd = qd - repmat(qmean,1,size(vIn,2)); 
% ymean = mean(y,2);
% y = y - repmat(ymean,1,size(q,2));

%GPR with exp() kernel
M = pdist2(q', q');
Md = pdist2(qd', q');
K = p(1) * exp(-p(2) * M.^2);
Kd = p(1) * exp(-p(2) * Md.^2);
invK = pinv(K + p(3) * eye(size(K))); 

%Output
yd = (Kd * invK * y')'; % + repmat(ymean,1,size(qd,2)); 

if nargout>1
	SigmaOut = zeros(size(yd,1), size(yd,1), size(yd,2));
	if covopt==0
		%Evaluate Sigma as in Rasmussen, 2006
		Mdd = pdist2(qd',qd');
		Kdd = p(1) * exp(-p(2) * Mdd.^2);
		S = Kdd - Kd * invK * Kd';
		for t=1:size(yd,2)
			SigmaOut(:,:,t) = eye(size(yd,1)) * S(t,t); 
		end
	else
		%Evaluate Sigma as in GMR
		%nbSamples = size(y,2) / size(yd,2);
		%yd = repmat(yd,1,nbSamples);
		for t=1:size(yd,2)
			W = diag(K(t,:) * invK);
			ym = repmat(yd(:,t), 1, size(y,2));
			%SigmaOut(:,:,t) = (y-yd) * W * (y-yd)' + eye(size(vOut,1))*diagRegularizationFactor;  
			SigmaOut(:,:,t) = (y-ym) * W * (y-ym)' + eye(size(yd,1))*diagRegularizationFactor; 
		end
	end
end
