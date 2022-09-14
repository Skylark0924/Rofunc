function demo_GaussProd4nullspace_3D01
% 3D illustration of using a product of Gaussians to compute the hierarchy of three tasks.
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
% }
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

addpath('./m_fcts/');


%% Set GMM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 3;

Mu1 = [10; 10; 0];
d1 = [0; 0; 1] .* 1E2;
Q1 = d1 * d1';
S1 = inv(Q1 + eye(nbVar) .* 1E-3);

Mu2 = [0; 0; 10];
d2 = [0; -2; 1] .* 1E2;
Q2 = d2 * d2';
S2 = inv(Q2 + eye(nbVar) .* 1E-3);

Mu3 = [-3; 5; 5];
d3 = [3; -1; 2] .* 1E0;
S3 = d3 * d3' + eye(nbVar) .* 1E0;
Q3 = inv(S3);


% %Working version (simple version, but requires ridge regression to be stable)
% J = Q1;
% pinvJw = S2 * J' / (J * S2 * J' + eye(nbVar)*1E-4); %1E-8 needs to be added as ridge regression coefficient 
% % pinvJw = S2 * J' * inv(J * S2 * J' + eye(nbVar)*1E-4); %1E-4 needs to be added as ridge regression coefficient 
% Nw = eye(nbVar) - pinvJw * J;
% x = pinvJw * (Q1 * Mu1) + Nw * Mu2;

%PoG
Q = Q1 + Q2 + Q3;
S = inv(Q + eye(nbVar)*1E-4);
Mu = S * (Q1 * Mu1 + Q2 * Mu2 + Q3 * Mu3);

Q12 = Q1 + Q2;
S12 = inv(Q12 + eye(nbVar)*1E-4);
Mu12 = S12 * (Q1 * Mu1 + Q2 * Mu2);

Q13 = Q1 + Q3;
S13 = inv(Q13 + eye(nbVar)*1E-4);
Mu13 = S13 * (Q1 * Mu1 + Q3 * Mu3);

Q23 = Q2 + Q3;
S23 = inv(Q23 + eye(nbVar)*1E-4);
Mu23 = S23 * (Q2 * Mu2 + Q3 * Mu3);

Q123 = Q12 + Q3;
S123 = inv(Q123 + eye(nbVar)*1E-4);
Mu123 = S123 * (Q12 * Mu12 + Q3 * Mu3);

% Q132 = Q13 + Q2;
% S132 = inv(Q132 + eye(nbVar)*1E-4);
% Mu132 = S132 * (Q13 * Mu13 + Q2 * Mu2);
% 
% Q231 = Q23 + Q1;
% S231 = inv(Q231 + eye(nbVar)*1E-4);
% Mu231 = S231 * (Q23 * Mu23 + Q1 * Mu1);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1200,800]); hold on; box on; rotate3d on; axis off; 

plot3(Mu1(1), Mu1(2), Mu1(3), '.','markersize',20,'color',[.8 0 0]);
plot3(Mu2(1), Mu2(2), Mu2(3), '.','markersize',20,'color',[0 .7 0]);
plot3(Mu3(1), Mu3(2), Mu3(3), '.','markersize',20,'color',[0 0 .8]);
plot3(Mu12(1), Mu12(2), Mu12(3), '.','markersize',20,'color',[.8 .7 0]);
plot3(Mu13(1), Mu13(2), Mu13(3), '.','markersize',20,'color',[.8 0 .8]);
plot3(Mu23(1), Mu23(2), Mu23(3), '.','markersize',20,'color',[0 .7 .8]);

plot3(Mu(1), Mu(2), Mu(3), '.','markersize',20,'color',[0 0 0]);
% plot3(Mu123(1), Mu123(2), Mu123(3), '.','markersize',20,'color',[.2 .2 .2]);
% plot3(Mu132(1), Mu132(2), Mu132(3), '.','markersize',20,'color',[.4 .4 .4]);
% plot3(Mu231(1), Mu231(2), Mu231(3), '.','markersize',20,'color',[.6 .6 .6]);

plotGMM3D(Mu1, S1, [.8 0 0], .2);
plotGMM3D(Mu2, S2, [0 .7 0], .2);
plotGMM3D(Mu3, S3, [0 0 .8], .2);
plotGMM3D(Mu12, S12+eye(nbVar).*4E-3, [.8 .7 0], .2);
plotGMM3D(Mu13, S13, [.8 0 .8], .2);
plotGMM3D(Mu23, S23, [0 .7 .8], .2);

plotGMM3D(Mu, S+eye(nbVar).*4E-3, [0 0 0], .4);
% plotGMM3D(Mu123, S123+eye(nbVar).*4E-3, [.2 .2 .2], .4);
% plotGMM3D(Mu132, S132+eye(nbVar).*4E-3, [.4 .4 .4], .4);
% plotGMM3D(Mu231, S231+eye(nbVar).*4E-3, [.6 .6 .6], .4);

% plot3(x(1), x(2), x(3), 'ko','markersize',8,'linewidth',2);
axis equal; axis([-1 1 -.5 1 -.2 1]*15); axis vis3d; view(140,25); 
%set(gca,'xtick',[],'ytick',[],'ztick',[]);

% print('-dpng','graphs/GaussProd4nullspace_3D01.png');
pause;
close all;