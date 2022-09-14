function h = plotArmBasis(p1, sz, facecol, edgecol, alpha)
% Display of the base of a planar robot arm.
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


nbSegm = 30;
sz = sz*1.2;

t1 = linspace(0,pi,nbSegm-2);
xTmp(1,:) = [sz*1.5 sz.*1.5*cos(t1) -sz*1.5];
xTmp(2,:) = [-sz*1.2 sz.*1.5*sin(t1) -sz*1.2];
xTmp(3,:) = zeros(1,nbSegm);
x = xTmp + repmat(p1,1,nbSegm);
h = patch(x(1,:),x(2,:),x(3,:),facecol,'edgecolor',edgecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha); 

xTmp2(1,:) = linspace(-sz*1.2,sz*1.2,5);
xTmp2(2,:) = repmat(-sz*1.2,1,5);
xTmp2(3,:) = zeros(1,5);
x2 = xTmp2 + repmat(p1,1,5);
x3 = xTmp2 + repmat(p1+[-0.5;-1;0]*sz,1,5);
for i=1:5
	if facecol==[1,1,1]
		h = [h patch([x2(1,i) x3(1,i)], [x2(2,i) x3(2,i)], edgecol,'edgecolor',edgecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha)]; 
	else
		h = [h patch([x2(1,i) x3(1,i)], [x2(2,i) x3(2,i)], facecol,'edgecolor',facecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha)]; 
	end
end
