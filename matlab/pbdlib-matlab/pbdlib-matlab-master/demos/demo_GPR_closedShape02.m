function demo_GPR_closedShape02
% Gaussian process implicit surface (GPIS) representation with thin-plate covariance function in GPR
% 
% If this code is useful for your research, please cite the related publications:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
% }
% @inproceedings{Williams07,
% 	author = "Williams, O. and Fitzgibbon, A.",
% 	title = "Gaussian Process Implicit Surfaces",
% 	booktitle = "Gaussian Processes in Practice",
% 	year = "2007"
% }
% 
% Copyright (c) 2019 y==1ap Research Institute, http://y==1ap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.y==1ap.ch/software/pbdlib/
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

% addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDataRepro = 50^2; %Number of datapoints in a reproduction
% nbVarX = 2; %Dimension of x
p = [1E0, 1E-5]; %Thin-plate covariance function parameters 


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x = [-3 -2 -1  0  1  2  3  3  3  2  1  1  1  0 -1 -1 -1 -2 -3 -3;
% 		  2  2  2  2  2  2  2  1  0  0  0 -1 -2 -2 -2 -1  0  0  0  1];
% x = [x, mean(x,2), [-4, -4, 4, 4; 4 -4 -4 4]] * 1E-1;
% x = x(:,6:end); %Simulate missing datapoints
% nbData = size(x,2);
% y = [zeros(1,nbData-5), 1, -1, -1, -1, -1]; %0, 1 and -1 represent border, interior, and exterior points
%mean(x,2)+[-1;0], mean(x,2)+[1;0], mean(x,2)+[0;-1], mean(x,2)+[0;1]

% x0 = [3  3  2   2  0 -1 -1 -3 -3;
% 		  2  1  0  -1 -2 -2 -1  2  1];
% x = [x0, mean(x0,2), [-3;3], [-2;-3], [4;-2], [-3;3], [4;2], [2;-2] [4;4]] * 1E-1; 
% y = [zeros(1,size(x0,2)), 1, -1, -1, -1, -1, -1, -1, -1]; %0, 1 and -1 represent border, interior, and exterior points

% x = [-.3, -.1, .1; 0, 0, 0];
% y = [-1, 0, 1]; %-1, 0 and 1 represent exterior, border and interior points

% x = rand(2,15) - .5; 
% y = randi(3,1,15) - 2; %-1, 0 and 1 represent exterior, border and interior points

% x = [-0.3, -0.1,  0.1,  0.0,  0.4,  0.4,  0.3,  0.0, -0.3,  0.0,  0.2,  0.0,  0.4; ...
%       0.0,  0.0,  0.0,  0.4, -0.4,  0.4,  0.1,  0.2,  0.3, -0.4, -0.2, -0.3, -0.2];
% y = [-1,    0,    1,   -1,   -1,   -1,    0,    0,   -1,   -1,    0,    0,    0];

% x = [-0.3, -0.2,  0.1, -0.2,  0.2,  0.3,  0.4; ...
%       0.0,  0.0,  0.0,  0.4, -0.2,  0.2, -0.4];
% y = [-1,    0,    1,   -1,    0     0,   -1];

x = [-0.4500   -0.0006    0.0829   -0.3480   -0.1983; ...
     -0.4500   -0.2000   -0.3920   -0.3451   -0.0423];
y = [-1     1     0     0     0];


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Xm, Ym] = meshgrid(linspace(-.5,.5,nbDataRepro^.5), linspace(-.5,.5,nbDataRepro^.5));
xs = [Xm(:)'; Ym(:)'];
p(1) = max(max(pdist2(xs', xs'))); %Refine thin-plate covariance function parameter

%Circle as geometric prior
rc = 4E-1; %Radius of circle
xc = [.05; .05]; %Location of circle
S = eye(2) * rc^-2;
MuS = .5 * rc * diag(1 - (xs-repmat(xc,1,nbDataRepro))' * S * (xs-repmat(xc,1,nbDataRepro)))';
Mu = .5 * rc * diag(1 - (x-repmat(xc,1,size(x,2)))' * S * (x-repmat(xc,1,size(x,2))))';

K = covFct(x, x, p, 1); %Inclusion of noise on the inputs for the computation of K
[Ks, dKs] = covFct(xs, x, p);
% ry = (Ks / K * y')'; %GPR with Mu=0
ry = MuS + (Ks / K * (y - Mu)')';
rdy = [(dKs(:,:,1) / K * (y - Mu)')'; (dKs(:,:,2) / K * (y - Mu)')']; %Gradients

%Redefine gradient
% a = min(ry,0); %Amplitude
% for t=1:size(rdy,2)
% 	rdy(:,t) = - exp(a(t)) * rdy(:,t) / norm(rdy(:,t)); %Vector moving away from contour, with stronger influence when close to the border 
% end

%Uncertainty evaluation
Kss = covFct(xs, xs, p); 
S = Kss - Ks / K * Ks';
% nbVarY = 1; %Dimension of y 
% SigmaY = eye(nbVarY);
% SigmaOut = zeros(nbVarY, nbVarY, nbDataRepro);
% for t=1:nbDataRepro
% 	SigmaOut(:,:,t) = SigmaY * S(t,t); 
% end

% %Generate stochastic samples from the prior 
% nbRepros = 6; %Number of randomly sampled reproductions
% [V,D] = eig(Kss);
% for n=2:nbRepros/2
% 	yp = real(V*D^.5) * randn(nbDataRepro, nbVarY) * SigmaY .* 2E-1; 
% 	r(n).y = yp';
% end
% 
% %Generate stochastic samples from the posterior 
% [V,D] = eig(S);
% for n=nbRepros/2+1:nbRepros
% 	ys = real(V*D^.5) * randn(nbDataRepro, nbVarY) * SigmaY .* 0.5 + ry(nbVarX+1:end,:)'; 
% 	r(n).y = ys';
% end


%% Spatial plots 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1200 800],'color',[1,1,1]); hold on; axis off; %'PaperPosition',[0 0 14 5]
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
set(gca,'LooseInset',[0,0,0,0]);
colormap(repmat(linspace(1,.4,64),3,1)');

%Plot center of GP (acting as prior if points on the contours are missing)
subplot(2,5,1); hold on; axis off; rotate3d on; title('Prior (circular shape)','fontsize',26);
% for n=2:nbRepros/2
% 	coltmp = [.5 .5 .5] + [.5 .5 .5].*rand(1);
% 	mesh(Xm, Ym, reshape(r(n).Data(3,:), nbDataRepro^.5, nbDataRepro^.5), 'facealpha',.4,'edgealpha',.4,'facecolor',coltmp,'edgecolor',coltmp); %Prior samples
% end
mesh(Xm, Ym, reshape(MuS, nbDataRepro^.5, nbDataRepro^.5), 'facealpha',.8,'edgealpha',.8,'edgecolor',[.7 .7 .7]);
mesh(Xm, Ym, zeros(nbDataRepro^.5, nbDataRepro^.5), 'facealpha',.3,'edgealpha',.6,'facecolor',[0 0 0],'edgecolor','none');
tl = linspace(0,2*pi,100);
plot3(xc(1)+cos(tl)*rc, xc(2)+sin(tl)*rc, zeros(1,100), '-','linewidth',2,'color',[0 0 0]);
view(3); axis vis3d;

%Plot posterior distribution (3D)
subplot(2,5,6); hold on; axis off; rotate3d on; title('Posterior','fontsize',26);
% for n=nbRepros/2+1:nbRepros
% 	coltmp = [.5 .5 .5] + [.5 .5 .5].*rand(1);
% 	mesh(Xm, Ym, reshape(r(n).Data(3,:), nbDataRepro.^.5, nbDataRepro.^.5), 'facealpha',.4,'edgealpha',.4,'facecolor',coltmp,'edgecolor',coltmp); %posterior samples
% end
mesh(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), 'facealpha',.8,'edgealpha',.8,'edgecolor',[.7 .7 .7]);
mesh(Xm, Ym, zeros(nbDataRepro^.5, nbDataRepro^.5), 'facealpha',.3,'edgealpha',.6,'facecolor',[0 0 0],'edgecolor','none');
contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [0,0], 'linewidth',2,'color',[0 0 0]); 
%plot3(x(1,y==1), x(2,y==1), y(y==1)+.04, '.','markersize',38,'color',[.8 0 0]); %Interior points
plot3(x(1,y==0), x(2,y==0), y(y==0)+.04, '.','markersize',38,'color',[.8 .4 0]); %Border points
%plot3(x(1,y==-1), x(2,y==-1), y(y==-1)+.04, '.','markersize',38,'color',[0 .6 0]); %Exterior points
view(3); axis vis3d;

%Plot posterior distribution (2D)
subplot(2,5,[2,3,7,8]); hold on; axis off; title('Distance to contour and gradient','fontsize',26);
mshbrd = [-.5 -.5 .5 .5 -.5; -.5 .5 .5 -.5 -.5];
plot(mshbrd(1,:), mshbrd(2,:), '-','linewidth',1,'color',[0 0 0]);
surface(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5)-max(ry), 'FaceColor','interp','EdgeColor','interp');
quiver(xs(1,1:4:end), xs(2,1:4:end), rdy(1,1:4:end), rdy(2,1:4:end),'color',[.2 .2 .2]);
% quiver(xs(1,ry>0), xs(2,ry>0), rdy(1,ry>0), rdy(2,ry>0), 'color',[.8 .2 .2]);
% quiver(xs(1,ry<0), xs(2,ry<0), rdy(1,ry<0), rdy(2,ry<0), 'color',[.2 .7 .2]);
%h(1) = plot(x(1,y==1), x(2,y==1), '.','markersize',48,'color',[.8 0 0]); %Interior points
h(1) = plot(x(1,y==0), x(2,y==0), '.','markersize',48,'color',[.8 .4 0]); %Border points
%h(3) = plot(x(1,y==-1), x(2,y==-1), '.','markersize',48,'color',[0 .6 0]); %Exterior points
h(2) = plot(xc(1)+cos(tl)*rc, xc(2)+sin(tl)*rc, '--','linewidth',2,'color',[0 0 0]);
[~,h(3)] = contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [0,0], 'linewidth',4,'color',[0 0 0]);
% contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [-.4,-.4], 'linewidth',2,'color',[.2 .2 .2]);
% contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [-.8,-.8], 'linewidth',2,'color',[.4 .4 .4]);
% contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [-1.2,-1.2], 'linewidth',2,'color',[.6 .6 .6]);
% patch(c(1,2:end), c(2,2:end), ones(1,size(c,2)-1), [.8 0 0], 'edgecolor','none','facealpha', .1); 
% contour(Xm, Ym, reshape(ry(3,:)+2E0.*diag(S).^.5', nbDataRepro.^.5, nbDataRepro.^.5), [0,0], 'linewidth',1,'color',[.6 .6 .6]); 
% contour(Xm, Ym, reshape(ry(3,:)-2E0.*diag(S).^.5', nbDataRepro.^.5, nbDataRepro.^.5), [0,0], 'linewidth',1,'color',[.6 .6 .6]); 
axis tight; axis equal; axis([-.5 .5 -.5 .5]); axis ij;
%legend(h,{'Demonstrated interior points (y=1)','Demonstrated contour points (y=0)','Demonstrated exterior points (y=-1)','Prior for contour estimation','Estimated contour'},'fontsize',16,'location','southwest');
legend(h,{'Provided contour points','Prior for contour estimation','Estimated contour'},'fontsize',26,'location','southwest');
% legend('boxoff');

%Plot uncertainty (2D)
subplot(2,5,[4,5,9,10]); hold on; axis off; title('Uncertainty','fontsize',26);
plot(mshbrd(1,:), mshbrd(2,:), '-','linewidth',1,'color',[0 0 0]);
% surface(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5)-max(ry), 'FaceColor','interp','EdgeColor','interp');
surface(Xm, Ym, reshape(diag(S), nbDataRepro^.5, nbDataRepro^.5)-max(S(:)), 'FaceColor','interp','EdgeColor','interp');
plot(x(1,y==1), x(2,y==1), '.','markersize',48,'color',[.8 0 0]); %Interior points
plot(x(1,y==0), x(2,y==0), '.','markersize',48,'color',[.8 .4 0]); %Border points
plot(x(1,y==-1), x(2,y==-1), '.','markersize',48,'color',[0 .6 0]); %Exterior points
contour(Xm, Ym, reshape(ry, nbDataRepro^.5, nbDataRepro^.5), [0,0], 'linewidth',4,'color',[0 0 0]);
plot(xc(1)+cos(tl)*rc, xc(2)+sin(tl)*rc, '--','linewidth',2,'color',[0 0 0]);
axis tight; axis equal; axis([-.5 .5 -.5 .5]); axis ij;

%print('-dpng','graphs/GPIS_new01.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
function d = substr(x1, x2)
	d = x1 - x2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
function [K, dK] = covFct(x1, x2, p, flag_noiseObs)
	if nargin<4
		flag_noiseObs = 0;
	end
	
% 	%Thin plate covariance function (for 3D implicit shape)
% 	K = 12^-1 * (2 * pdist2(x1',x2').^3 - 3 * p(1) * pdist2(x1',x2').^2 + p(1)^3); %Kernel
% 	dK(:,:,1) = 12^-1 * (6 * pdist2(x1',x2') .* substr(x1(1,:)',x2(1,:)) - 6 * p(1) * substr(x1(1,:)',x2(1,:))); %Derivatives along x1
% 	dK(:,:,2) = 12^-1 * (6 * pdist2(x1',x2') .* substr(x1(2,:)',x2(2,:)) - 6 * p(1) * substr(x1(2,:)',x2(2,:))); %Derivatives along x2
% % 	for i=1:size(x1,2)
% % 		for j=1:size(x2,2)
% % 			e = x1(:,i) - x2(:,j);
% % % 			K(i,j) = 12^-1 * (2 * pdist2(x1(:,i)',x2(:,j)')^3 - 3 * p(1) * pdist2(x1(:,i)',x2(:,j)')^2 + p(1)^3);
% % % 			K(i,j) = 12^-1 * (2 * norm(e)^3 - 3 * p(1) * e'*e + p(1)^3);
% % 			K(i,j) = 12^-1 * (2 * (e'*e)^1.5 - 3 * p(1) * e'*e + p(1)^3); %Kernel (slow one by one computation)
% % 			dK(i,j,:) = 12^-1 * (6 * (e'*e)^.5 * e  - 6 * p(1) * e); %Derivatives (slow one by one computation)
% % 		end
% % 	end


% 	%Thin plate covariance function (for 2D implicit shape -> does not seem to work)
% 	K = 2 * pdist2(x1',x2').^2 .* log(pdist2(x1',x2')) - (1 + 2*log(p(1))) .* pdist2(x1',x2').^2 + p(1)^2;


	%RBF covariance function
	p = [5E-2^-1, 1E-4, 1E-2];
	K = p(3) * exp(-p(1) * pdist2(x1', x2').^2); %Kernel
	dK(:,:,1) = -p(1) * p(3) * exp(-p(1) * pdist2(x1', x2').^2) .* substr(x1(1,:)',x2(1,:)); %Derivatives along x1
	dK(:,:,2) = -p(1) * p(3) * exp(-p(1) * pdist2(x1', x2').^2) .* substr(x1(2,:)',x2(2,:)); %Derivatives along x2
% 	for i=1:size(x1,2)
% 		for j=1:size(x2,2)
% 			K(i,j) = p(3) * exp(-p(1) * pdist2(x1(:,i)', x2(:,j)').^2); %Kernel (slow one by one computation)
% 			dK(i,j,:) = -p(1) * p(3) * exp(-p(1) * pdist2(x1(:,i)', x2(:,j)').^2) * (x1(:,i) - x2(:,j)); %Derivatives (slow one by one computation)
% 		end
% 	end
	
	if flag_noiseObs==1
		K = K + p(2) * eye(size(x1,2),size(x2,2)); %Consy==-1ration of noisy observation y
	end
end
