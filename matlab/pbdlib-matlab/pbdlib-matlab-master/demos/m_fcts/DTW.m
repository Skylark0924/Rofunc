function [x2, y2, p] = DTW(x, y, w)
% Trajectory realignment through dynamic time warping.
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


if nargin<3
  w = Inf;
end

sx = size(x,2);
sy = size(y,2);

w = max(w,abs(sx-sy)); 

%Initialization
D = ones(sx+1,sy+1) * Inf; 
D(1,1) = 0;

%DP loop
for i=1:sx
  for j=max(i-w,1):min(i+w,sy)
    D(i+1,j+1) = norm(x(:,i)-y(:,j)) + min([D(i,j+1), D(i+1,j), D(i,j)]);
  end
end

i=sx+1; j=sy+1;
p=[];
while i>1 && j>1
 [~,id] = min([D(i,j-1), D(i-1,j), D(i-1,j-1)]);
 if id==1
   j=j-1;
 elseif id==2
   i=i-1;
 else
   i=i-1;
   j=j-1;
 end
 p = [p [i;j]];
end

p = fliplr(p(:,1:end-1)-1);

x2 = x(:,p(1,:));
y2 = y(:,p(2,:));

%Resampling
x2 = spline(1:size(x2,2), x2, linspace(1,size(x2,2),sx));
y2 = spline(1:size(y2,2), y2, linspace(1,size(y2,2),sx));
