function demo_covariance01
% Covariance computation in various forms
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 3; %Dimension of datapoint
nbData = 100; %Number of datapoints

x = rand(nbVar,nbData);
xm = mean(x,2); %average
xc = x - repmat(xm,1,nbData); %centered data

%% Covariance computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Version 1 (standard)
S1 = cov(x')

%Version 2 (in matrix form with centered data)
S2 = xc * xc' ./ (nbData-1)

%Version 3 (as weighted sum of singular covariances)
w = 1/(nbData-1);
S3 = zeros(nbVar);
for t=1:nbData
	S3 = S3 + w * xc(:,t) * xc(:,t)';
end
S3

%Version 4 (with centering matrix)
C = eye(nbData) - ones(nbData,1) * ones(1,nbData) .* 1/nbData; %xc=x*C
S4 = x * (C * C') * x' ./ (nbData-1)
% S = (eye(nbData) - ones(nbData)/nbData) / (nbData-1);
% S4 = x * S * x'

%Version 5 (no explicit recentering)
S5 = zeros(nbVar);
for i=1:nbData
	for j=1:nbData
		S5 = S5 + (x(:,i)-x(:,j)) * (x(:,i)-x(:,j))';
	end
end
S5 = S5 ./ (2*nbData*(nbData-1))

