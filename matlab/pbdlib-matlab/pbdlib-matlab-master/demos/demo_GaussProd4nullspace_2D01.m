function demo_GaussProd4nullspace_2D01
% 2D illustration of using a product of Gaussians to compute the hierarchy of two tasks
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
Mu1 = zeros(2,1);
% Q1 = [0 0; 0 1E3]
d = [.5; 1.2] .* 1E1;
Q1 = d * d' + eye(2).*1E-1;

% inv(Q1+eye(2)*1E-12)
% pinv(Q1+eye(2)*1E-12)
% [SJ,VJ] = eig(Q1);
% SJ(SJ>0) = SJ(SJ>0).^-1;
% S1 = VJ * SJ * VJ'
S1 = inv(Q1);

Mu2 = [1; 2];
d2 = [.05; -.2] .* 8;
S2 = d2 * d2' + eye(2) * 1E-1;
Q2 = inv(S2);

%Version 1 (simple version, but requires ridge regression to be stable)
J = Q1;
pinvJw = S2 * J' / (J * S2 * J' + eye(2)*1E-6); %1E-6 needs to be added as ridge regression coefficient 
% pinvJw = S2 * J' * inv(J * S2 * J' + eye(2)*1E-4); %1E-4 needs to be added as ridge regression coefficient 
Nw = eye(2) - pinvJw * J;
x = pinvJw * (Q1 * Mu1) + Nw * Mu2;

% %Version 2 (with Q1 and S2)
% % U1 = sqrtm(Q1);
% % U2 = sqrtm(S2);
% [V1,D1] = eig(Q1);
% U1 = V1 * D1.^.5; 
% [V2,D2] = eig(S2);
% U2 = V2 * D2.^.5; 
% J =  U1' * U2;
% % pinvJ = J' / (J * J' + eye(2)*1E-6);
% pinvJ = pinv(J);
% N = eye(2) - pinvJ * J;
% x = U2 * (pinvJ * (U1' * Mu1) + N * (U2 \ Mu2));

% %Version 3 (with Q1 and Q2)
% % U1 = sqrtm(Q1);
% % U2 = sqrtm(Q2);
% [V1,D1] = eig(Q1);
% U1 = V1 * D1.^.5; 
% [V2,D2] = eig(Q2);
% U2 = V2 * D2.^.5; 
% J =  U1' * inv(U2');
% % pinvJ = J' / (J * J' + eye(2)*1E-6);
% pinvJ = pinv(J);
% N = eye(2) - pinvJ * J;
% x = inv(U2') * (pinvJ * (U1' * Mu1) + N * (U2' * Mu2));

% %Version 4
% [V2,D2] = eig(S2);
% U2 = V2 * D2.^.5;
% % U2 = sqrtm(S2);
% J =  Q1 * U2;
% % pinvJ = J' / (J * J' + eye(2)*1E-8);
% pinvJ = pinv(J);
% N = eye(2) - U2 * pinvJ * Q1;
% x = U2 * pinvJ * (Q1 * Mu1) + N * Mu2;


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1200,800]); hold on; box on; %axis off;
plotGMM(Mu1, S1, [.8 0 0],.3);
plotGMM(Mu2, S2, [0 .8 0],.3);
plot(x(1), x(2), 'ko','markersize',8,'linewidth',2);
axis equal; axis([-0.5 2.5 -0.6 1.6]*3);
set(gca,'xtick',[],'ytick',[]);

%print('-dpng','graphs/demo_GaussProd4nullspace_2D01.png');
pause;
close all;