function Xout = plotDistrib1D(Pt, boundbox, color, valAlpha, orient_option)
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
	boundbox = [-1,1,0,1];
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
	Xout = Pt / max(Pt);
	P = [Xin; boundbox(2)+boundbox(4)*Xout]; 
	patch([P(1,1) P(1,:) P(1,end)], [boundbox(2) P(2,:) boundbox(2)], ...
		color, 'linestyle', ':', 'lineWidth', 2, 'EdgeColor', darkcolor, 'facealpha', valAlpha*0,'edgealpha', valAlpha);
else
	Xin(1,:) = linspace(boundbox(2), boundbox(2)+boundbox(4), nbData);
	Xout = Pt / max(Pt);
	P = [boundbox(1)+boundbox(3)*Xout; Xin]; 
	patch([boundbox(1) P(1,:) boundbox(1)], [P(2,1) P(2,:) P(2,end)], ...
		color, 'linestyle', ':', 'lineWidth', 2, 'EdgeColor', darkcolor, 'facealpha', valAlpha*0,'edgealpha', valAlpha);
end
