function demo_Riemannian_S1_interp01
% Interpolation on 1-sphere manifold (formulation with complex numbers a+b*i)
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
% Written by Sylvain Calinon, https://calinon.ch/
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

addpath('./m_fcts/');


%% Generate datapoints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = exp(1i * (rand(1,2)-0.5) * 2*pi);


%% Geodesic interpolation (interpolation between two points can be computed in closed form)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 50; %Number of interpolation steps
w = linspace(0,1,nbData);
xi = zeros(1,nbData);
for t=1:nbData
	xi(t) = expmap(w(t) * logmap(x(2),x(1)), x(1));
end


%% 2D plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800]); hold on; axis off; 
t = linspace(0, 2*pi, 100);
plot(cos(t), sin(t), '-','color',[.7 .7 .7]);
plot(0,0,'k+');
plot(real(xi), imag(xi), '-','linewidth',2,'color',[0 0 0]);
plot(real(x), imag(x), '.','markersize',20,'color',[.8 0 0]);
u = logmap(x(2), x(1));
x = [real(x(1)); imag(x(1))];
xn = [-x(2); x(1)];
xn = xn .* u ./ norm(xn);
plot2DArrow(x, xn, [.8,0,0], 2, .04);
axis equal; 

% print('-dpng','graphs/demo_Riemannian_S1_interp01.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, x0)
	x = x0 * exp(u*1i);
end

function u = logmap(x, x0)
	u = imag(log(x0' * x));
end