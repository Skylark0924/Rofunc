function [P,h] = plotGaussian1D(Mu, Sigma, boundbox, color, valAlpha, orient_option)
% Plot Gaussian profile horizontally or vertically
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
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
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

if nargin<3
	boundbox = [-1,0,2,1];
end
if nargin<4
	color = [0,0,0];
end
if nargin<5
	valAlpha = .4;
end
if nargin<6
	orient_option = 'h';
end

nbData = 400; 
darkcolor = color * 0.5;

if orient_option=='h'
	Xin(1,:) = linspace(boundbox(1), boundbox(1)+boundbox(3), nbData);
	X = [Xin; gaussPDF(Xin, Mu(1), Sigma(1,1))];
	Xout = sum(reshape(X(2,:),nbData,1),2)';
	Xout = Xout / max(Xout);
	P = [Xin; boundbox(2)+boundbox(4)*Xout]; 
	h(1) = patch([P(1,1) P(1,:) P(1,end)], [boundbox(2) P(2,:) boundbox(2)], ...
		color, 'lineWidth', 1, 'EdgeColor', darkcolor, 'facealpha', valAlpha,'edgealpha', valAlpha);
	h(2) = plot([Mu(1) Mu(1)], [boundbox(2) max(P(2,:))], '-','linewidth',1,'color',min(darkcolor+0.3,1));
else
	Xin(1,:) = linspace(boundbox(2), boundbox(2)+boundbox(4), nbData);
	X = [gaussPDF(Xin, Mu(1), Sigma(1,1)); Xin];
	Xout = sum(reshape(X(1,:),nbData,1),2)';
	Xout = Xout / max(Xout);
	P = [boundbox(1)+boundbox(3)*Xout; Xin]; 
	h(1) = patch([boundbox(1) P(1,:) boundbox(1)], [P(2,1) P(2,:) P(2,end)], ...
		color, 'lineWidth', 1, 'EdgeColor', darkcolor, 'facealpha', valAlpha,'edgealpha', valAlpha);
	h(2) = plot([boundbox(1) max(P(1,:))], [Mu(1) Mu(1)], '-','linewidth',1,'color',min(darkcolor+0.3,1));
end
