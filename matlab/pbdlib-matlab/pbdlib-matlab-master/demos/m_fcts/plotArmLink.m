function [p2, h] = plotArmLink(a1, d1, p1, sz, facecol, edgecol, alpha)
% Display of a link of a planar robot arm.
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

p1 = p1 + [0; 0; .1];
t1 = linspace(0,-pi,nbSegm/2);
t2 = linspace(pi,0,nbSegm/2);
xTmp(1,:) = [sz.*sin(t1) d1+sz.*sin(t2)];
xTmp(2,:) = [sz.*cos(t1) sz.*cos(t2)];
xTmp(3,:) = zeros(1,nbSegm);
R = [cos(a1) -sin(a1) 0; sin(a1) cos(a1) 0; 0 0 0];
x = R*xTmp + repmat(p1,1,nbSegm);
p2 = R*[d1;0;0] + p1;
h = patch(x(1,:),x(2,:),x(3,:),facecol,'edgecolor',edgecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha); 
msh = [sin(linspace(0,2*pi,nbSegm)); cos(linspace(0,2*pi,nbSegm))]*sz*0.4;
h = [h patch(msh(1,:)+p1(1), msh(2,:)+p1(2), repmat(p1(3),1,nbSegm), facecol,'edgeColor',edgecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha)]; 
h = [h patch(msh(1,:)+p2(1), msh(2,:)+p2(2), repmat(p2(3),1,nbSegm), facecol,'edgeColor',edgecol,'linewidth',3,'edgealpha',alpha,'facealpha',alpha)]; 
