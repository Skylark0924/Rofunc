function demo_GMR_wrapped01
% Periodic motion encoded with wrapped GMM and reproduced with wrapped GMR in 2D (i.e., the first dimension is periodic)
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Noemie Jaquier and Sylvain Calinon
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

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 3; %Number of variables [t,x1]
nbData = 50; %Length of each trajectory
nbSamples = 4;

%Number of adjacent periods considered (high number -> more precise, but slower)
nbLim = [1, 0, 0]; %0 for non periodic dimension
%Maximum range value of each dimension
rgMax = [2*pi, 0, 0]; %0 for non periodic dimension
%List of range offsets to be tested (0 for non-periodic variable)
for j=1:model.nbVar
	v(j).rg = [-nbLim(j):nbLim(j)] * rgMax(j); 
end


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt = linspace(0,2*pi,5);
Data=[];
for n=1:nbSamples
	pts = [tt; sin(tt)*0.5 + sin(tt*3).^2*0.2 + 1.5];
	pts(2,:) = pts(2,:)*0.1 + [n n 3*n 2*n n+1]*0.01;
	Data = [Data spline(1:size(pts,2), pts, linspace(1,size(pts,2), nbData))]; %Resampling
end
Data(3,:) = Data(2,:);
%Check if the periodic variables lie in the predefined range
for i=1:model.nbVar
	if nbLim(i)>0
		idList = Data(i,:)>rgMax(i);
		Data(i,idList) = Data(i,idList) - rgMax(i);
	end
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data(:,1:nbData), model);
model = init_GMM_timeBased(Data(:,1:nbData), model);
model = EM_WGMM(Data, model, v);


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[DataOut, SigmaOut] = WGMR(model, Data(1,1:nbData), v(1), 1, 2:model.nbVar);


%% Plot parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmin = min(Data(2,:))*0.9;
rmax = max(Data(2,:))*1.1;
nbGrid = [13,3];
[th,r] = meshgrid(linspace(0,2*pi,nbGrid(1)),linspace(rmin,rmax,nbGrid(2)));
[xg,yg] = pol2cart(th,r);
nbGrid = [50,3];
[th2,r2] = meshgrid(linspace(0,2*pi,nbGrid(1)),linspace(rmin,rmax,nbGrid(2)));
[xg2,yg2] = pol2cart(th2,r2);
[xt(1,:),xt(2,:)] = pol2cart(0:pi/2:3*pi/2, repmat(rmax*1.1,1,4));


%% Plot 2D (polar and Cartesian)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 12 4],'position',[10,10,2300,900],'color',[1 1 1]); 
% set(0,'DefaultAxesLooseInset',[0,0,0,0]);

%Plot WGMM (polar plot)
%subplot(1,3,1); hold on; grid off; box off; axis off; %title('WGMM');
subaxis(1,3,1,'SpacingHorizontal',0.08); hold on; grid off; box off; axis off; 
%Plot polar grid
plot(xg,yg,'-','color',[.8 .8 .8]);
plot(xg2',yg2','-','color',[.8 .8 .8]);
%Plot model
plotWGMM(model.Mu, model.Sigma, [.8 0 0]);
%Plot demonstration data
[xx,yy] = pol2cart(Data(1,:),Data(2,:));
plot(xx,yy,'-','linewidth',1,'color',[.2 .2 .2]);
%Plot axes labels
lbl={'$0$','$\frac{\pi}{2}$','$\pi$','$\frac{3\pi}{2}$'};
for i=1:4
	text(xt(1,i),xt(2,i),lbl{i},'interpreter','latex','HorizontalAlignment','center','fontsize',16);
end
axis square; axis equal; 

%Plot WGMR (polar plot)
%subplot(1,3,2); hold on; grid off; box off; axis off; %title('WGMR');
subaxis(1,3,2,'SpacingHorizontal',0.08); hold on; grid off; box off; axis off; 
%Plot polar grid
plot(xg,yg,'-','color',[.8 .8 .8]);
plot(xg2',yg2','-','color',[.8 .8 .8]);
%Plot WGMR
X = [[Data(1,1:nbData) Data(1,1:nbData)]; [DataOut(1,:)-squeeze(SigmaOut(1,1,:).^.5)', flipud(DataOut(1,:)+squeeze(SigmaOut(1,1,:).^.5)')]];
[xx,yy] = pol2cart(X(1,:), X(2,:));
patch(xx, yy, min([0 .8 0]+0.85,1), 'lineWidth', 1, 'EdgeColor', 'none');
plot(xx(1:nbData),yy(1:nbData),'-','linewidth',1,'color',[0 .8 0]);
plot(xx(nbData+1:end),yy(nbData+1:end),'-','linewidth',1,'color',[0 .8 0]);
%Plor retrieved data
[xx,yy] = pol2cart(Data(1,1:nbData), DataOut(1,:));
plot(xx,yy,'-','linewidth',2,'color',[0 .4 0]);
%Plot axes labels
lbl={'$0$','$\frac{\pi}{2}$','$\pi$','$\frac{3\pi}{2}$'};
for i=1:4
	text(xt(1,i),xt(2,i),lbl{i},'interpreter','latex','HorizontalAlignment','center','fontsize',16);
end
axis square; axis equal; 

%Plot WGMM (Cartesian plot)
%subplot(1,3,3); hold on; grid on; box off; %title('WGMM (Cartesian)');
subaxis(1,3,3,'SpacingHorizontal',0.08); hold on; grid on; box off;  
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.8 0 0]);
for j=1:model.nbVar
	nr(j) = length(v(j).rg);
end
Xtmp = computeHypergrid_WGMM(v, size(Data,2));
DataGrid = repmat(Data, 1, prod(nr)) - Xtmp;
plot(DataGrid(1,:), DataGrid(2,:), '.', 'color', [.8 .8 .8]);
plot(Data(1,:), Data(2,:), '.', 'color', [0 0 0]);
plot(Data(1,[1,nbData]), DataOut(1,[1,nbData]), '-', 'linewidth', 2, 'color', [.4 .4 1]);
plot(Data(1,1:nbData), DataOut(1,:), '-', 'linewidth', 2, 'color', [.2 .8 .2]);
axis([min(DataGrid(1,:)), max(DataGrid(1,:)), rmin, rmax]); axis square; 
xlabel('t'); ylabel('x_1');
lbl={'0','','\pi','','2\pi'};
set(gca,'xtick',[0, pi/2, pi, 3*pi/2, 2*pi],'xticklabel',lbl,'ytick',r(:,1),'yticklabel',[]); %,'interpreter','latex'

%print('-dpng','-r300','graphs/WGMR01.png');

%Figure export from GNU Octave (then use 'pdflatex ..' or 'latex ..' then 'dvips -E .. -o ...eps')
%print('-depslatexstandalone','2d-circ-spac01.tex');
pause;
close all;
end