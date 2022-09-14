function [h, X] = plotGMM(Mu, Sigma, color, valAlpha)
% This function displays the parameters of a Gaussian Mixture Model (GMM).
% Inputs -----------------------------------------------------------------
%   o Mu:           D x K array representing the centers of K Gaussians.
%   o Sigma:        D x D x K array representing the covariance matrices of K Gaussians.
%   o color:        3 x 1 array representing the RGB color to use for the display.
%   o valAlpha:     transparency factor (optional).
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


nbStates = size(Mu,2);
nbDrawingSeg = 100;
darkcolor = color * .7; %max(color-0.5,0);
t = linspace(-pi, pi, nbDrawingSeg);
if nargin<4
	valAlpha = 1;
end

h = [];
X = zeros(2,nbDrawingSeg,nbStates);
for i=1:nbStates
	[V,D] = eig(Sigma(:,:,i));
	R = real(V*D.^.5);
% 	R = chol(Sigma(:,:,i))';
% 	R = sqrtm(Sigma(:,:,i));
	X(:,:,i) = R * [cos(t); sin(t)] + repmat(Mu(:,i), 1, nbDrawingSeg);
	if nargin>3 %Plot with alpha transparency
		h = [h patch(X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
		%MuTmp = [cos(t); sin(t)] * 0.3 + repmat(Mu(:,i),1,nbDrawingSeg);
		%h = [h patch(MuTmp(1,:), MuTmp(2,:), darkcolor, 'LineStyle', 'none', 'facealpha', valAlpha)];
		h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 10, 'color', darkcolor)];
	else %Plot without transparency
		%Standard plot
		h = [h patch(X(1,:,i), X(2,:,i), color, 'lineWidth', 1, 'EdgeColor', darkcolor)];
		h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 10, 'color', darkcolor)];
% 		%Plot only contours
% 		h = [h plot(X(1,:,i), X(2,:,i), '-', 'color', color, 'lineWidth', 1)];
	end
end
