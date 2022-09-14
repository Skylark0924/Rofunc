function demo_Riemannian_SPD_vecTransp01
% Verification of angle conservation in parallel transport on the symmetric positive definite (SPD) manifold S²+
% (covariance matrices of dimension 2x2)
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

addpath('./m_fcts');


%% Generate SPD data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = randn(2,10);
h = x*x';
x = randn(2,10);
g = x*x';
x = randn(2,10);
U0 = x*x';


%% Parallel transport of U0 from g to h
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(0,1,20);

for n=1:20
	t = tl(n);
	
	hist(n).MuMan = expmap(logmap(h,g)*t, g);
	
	Ac = transp(g, hist(n).MuMan);
	hist(n).U = Ac * U0 * Ac';
	  
	%Direction of the geodesic
	hist(n).dirG = logmap(h, hist(n).MuMan);
	if norm(hist(n).dirG) > 1E-5
		
		%Normalise the direction
		%hist(n).dirG = hist(n).dirG ./ norm(hist(n).dirG); %for S^3 manifold
		innormdir = trace(hist(n).MuMan^-.5 * hist(n).dirG * hist(n).MuMan^-1 * hist(n).dirG * hist(n).MuMan^-.5);
		hist(n).dirG = hist(n).dirG ./ sqrt(innormdir);

		%Compute the inner product with the first eigenvector
		%inprod(n) = hist(n).dirG' * hist(n).U(:,1); %for S^3 manifold
		inprod(n) = trace(hist(n).MuMan^-.5 * hist(n).dirG * hist(n).MuMan^-1 * hist(n).U * hist(n).MuMan^-.5);
	end
    
end
inprod

end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
	X = S^.5 * expm(S^-.5 * U * S^-.5) * S^.5;
end

function U = logmap(X,S)
% Logarithm map 
N = size(X,3);
for n = 1:N
% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
% 	U(:,:,n) = S * logm(S\X(:,:,n));
	[v,d] = eig(S\X(:,:,n));
	U(:,:,n) = S * v*diag(log(diag(d)))*v^-1;
end
end

function x = expmap_vec(u,s)
	nbData = size(u,2);
	d = size(u,1)^.5;
	U = reshape(u, [d, d, nbData]);
	S = reshape(s, [d, d]);
	x = zeros(d^2, nbData);
	for t=1:nbData
		x(:,t) = reshape(expmap(U(:,:,t),S), [d^2, 1]);
	end
end

function u = logmap_vec(x,s)
	nbData = size(x,2);
	d = size(x,1)^.5;
	X = reshape(x, [d, d, nbData]);
	S = reshape(s, [d, d]);
	u = zeros(d^2, nbData);
	for t=1:nbData
		u(:,t) = reshape(logmap(X(:,:,t),S), [d^2, 1]);
	end
end

function Ac = transp(S1,S2)
% 	t = 1;
% 	U = logmap(S2,S1);
% 	Ac = S1^.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^-.5;
	%Computationally economic way: 
	Ac = (S2/S1)^.5;
end