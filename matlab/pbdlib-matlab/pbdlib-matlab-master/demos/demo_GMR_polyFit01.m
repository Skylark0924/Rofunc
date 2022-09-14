function demo_GMR_polyFit01
% Polynomial fitting of handwriting motion with multivariate GMR
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 4; %Number of states in the GMM
model.nbVarIn = 3; %Dimension of input vector
model.nbVarOut = 2; %Dimension of output vector
model.nbVar = model.nbVarIn + model.nbVarOut; %Number of variables (input+output)
nbData = 100; %Length of a trajectory
nbSamples = 5; %Number of demonstrations


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/B.mat');
Data = [];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data, s(n).Data]; 
end

tIn = repmat(linspace(1,nbData,nbData),1,nbSamples);
DataIn = [];
for d=1:model.nbVarIn
	DataIn = [DataIn; tIn.^d]; %-> DataIn = [t; t.^2; t.^3; ...]
end
DataPol = [DataIn; Data]; % + randn(model.nbVar,size(X,1))*1E-15;

%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(DataPol, model, nbSamples);
model = EM_GMM(DataPol, model);

%[DataOut, SigmaOut, H] = GMR(model, DataIn, 1:model.nbVarIn, model.nbVarIn+1:model.nbVar);

%GMR
in = 1:model.nbVarIn;
out = model.nbVarIn+1:model.nbVar;
MuTmp = zeros(model.nbVarOut, model.nbStates);
tIn = tIn(:,1:nbData);
DataIn = DataIn(:,1:nbData);
DataOut = zeros(model.nbVarOut, nbData);
SigmaOut = zeros(model.nbVarOut, model.nbVarOut, nbData);
%Compute activation weight
for i=1:model.nbStates
	H(i,:) = model.Priors(i) * gaussPDF(DataIn(1,:), model.Mu(1,i), model.Sigma(1,1,i)); %H can be computed based on t only
end
H = H ./ repmat(sum(H)+realmin, model.nbStates, 1);
for t=1:nbData
	%Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i) / model.Sigma(in,in,i) * (DataIn(:,t)-model.Mu(in,i));
		DataOut(:,t) = DataOut(:,t) + H(i,t) * MuTmp(:,i);
	end
	%Compute conditional covariances
	for i=1:model.nbStates
		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i) / model.Sigma(in,in,i) * model.Sigma(in,out,i);
		SigmaOut(:,:,t) = SigmaOut(:,:,t) + H(i,t) * (SigmaTmp + MuTmp(:,i)*MuTmp(:,i)');
	end
	SigmaOut(:,:,t) = SigmaOut(:,:,t) - DataOut(:,t)*DataOut(:,t)' + eye(model.nbVarOut) * model.params_diagRegFact; 
end


% %% Plots
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,1000,800]); 
% for m=1:2
% 	subplot(2,1,m); hold on; %axis off; 
% 	plotGMM(model.Mu([1,model.nbVarIn+m],:), model.Sigma([1,model.nbVarIn+m],[1,model.nbVarIn+m],:), [.8 .8 .8]);
% 	for n=1:nbSamples
% 		plot(DataIn(1,:),s(n).Data(m,:),'-','linewidth',1,'color',[.2 .2 .2]);
% 	end
% 	plot(DataIn(1,:),DataOut(m,:),'-','linewidth',3,'color',[.8 0 0]);
% 	xlabel('x_1'); ylabel('y_1');
% 	axis([min(DataIn(1,:)), max(DataIn(1,:)), min(DataOut(m,:))-1, max(DataOut(m,:))+1]);
% end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 16 4],'position',[10,10,2300,800],'color',[1 1 1]); 
xx = round(linspace(1,64,model.nbStates));
clrmap = colormap('jet')*0.5;
clrmap = min(clrmap(xx,:),.9);
nbDrawingSeg = 100;
t = linspace(-pi, pi, nbDrawingSeg);	

%Compute polynomial segments
%[~,id] = max(H,[],1);
for i=1:model.nbStates
% 	DataInTmp = [];
% 	for d=1:model.nbVarIn 
% 		DataInTmp = [DataInTmp; tIn(id==i).^d]; 
% 	end
% 	MuDisp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i)/model.Sigma(in,in,i) * (DataInTmp - repmat(model.Mu(in,i),1,sum(id==i)));
	
	idDisp(i,:) = H(i,1:nbData) > .8./model.nbStates;
	size(idDisp(i,:))
	DataInTmp = [];
	for d=1:model.nbVarIn 
		DataInTmp = [DataInTmp; tIn(idDisp(i,:)).^d]; 
	end
	disp(i).Mu = model.Mu(out,i) + model.Sigma(out,in,i)/model.Sigma(in,in,i) * (DataInTmp - repmat(model.Mu(in,i),1,sum(idDisp(i,:))));
end	

%Spatial plot
axes('Position',[0 0 .2 1]); hold on; axis off;
plotGMM(DataOut, SigmaOut, [0 .8 0], .05);
for n=1:nbSamples
	plot(s(n).Data(1,:), s(n).Data(2,:), '-','linewidth',1,'color',[.7 .7 .7]);
end
for i=1:model.nbStates
	plot(disp(i).Mu(1,:), disp(i).Mu(2,:), '-','linewidth',2,'color',min(clrmap(i,:)+0.3,1));
end
for i=1:model.nbStates
	plotGMM(model.Mu(end-1:end,i), model.Sigma(end-1:end,end-1:end,i), min(clrmap(i,:)+0.5,1), .4);
end
plot(DataOut(1,:),DataOut(2,:),'-','markersize',16,'linewidth',2,'color',[0 .7 0]);
axis square; axis equal; 

%Timeline plot 
axes('Position',[.25 .58 .7 .4]); hold on; 
patch([DataIn(1,:), DataIn(1,end:-1:1)], [DataOut(1,:)+squeeze(SigmaOut(1,1,:).^.5)', DataOut(1,end:-1:1)-squeeze(SigmaOut(1,1,end:-1:1).^.5)'], [.2 .9 .2],'edgecolor','none','facealpha',.2);
for i=1:model.nbStates
% 	[V,D] = eig(model.Sigma([1,end-1],[1,end-1],i));
% 	R = real(V*D.^.5);
% 	msh = R * [cos(t); sin(t)] + repmat(model.Mu([1,end-1],i), 1, nbDrawingSeg);
% 	patch(msh(1,:), msh(2,:), min(clrmap(i,:)+0.5,1), 'lineWidth', 1, 'EdgeColor', min(clrmap(i,:)+0.5,1), 'facealpha', .4,'edgealpha', .4);
% 	[V,D] = eig(model.Sigma([2,end-1],[2,end-1],i));
% 	R = real(V*D.^.5);
% 	msh = R * [cos(t); sin(t)] + repmat(model.Mu([2,end-1],i), 1, nbDrawingSeg);
% 	patch(msh(1,:).^(1/2), msh(2,:), min(clrmap(i,:)+0.5,1), 'lineWidth', 1, 'EdgeColor', min(clrmap(i,:)+0.5,1), 'facealpha', .4,'edgealpha', .4);
% 	[V,D] = eig(model.Sigma([3,end-1],[3,end-1],i));
% 	R = real(V*D.^.5);
% 	msh = R * [cos(t); sin(t)] + repmat(model.Mu([3,end-1],i), 1, nbDrawingSeg);
% 	patch(msh(1,:).^(1/3), msh(2,:), min(clrmap(i,:)+0.5,1), 'lineWidth', 1, 'EdgeColor', min(clrmap(i,:)+0.5,1), 'facealpha', .4,'edgealpha', .4);
	plotGMM(model.Mu([1,end-1],i), model.Sigma([1,end-1],[1,end-1],i), min(clrmap(i,:)+0.5,1), .4);
end
for n=1:nbSamples
	plot(tIn, Data(1,(n-1)*nbData+1:n*nbData), '-','linewidth',1,'color',[.7 .7 .7]);
end
in = 1:model.nbVarIn; 
out = model.nbVarIn+1:model.nbVar;
for i=1:model.nbStates
% 	MuTmp = model.Mu(out(1),i) + model.Sigma(out(1),in,i)/model.Sigma(in,in,i) * (DataIn - repmat(model.Mu(in,i),1,nbData));
% 	plot(tIn, MuTmp, '.','linewidth',6,'markersize',26,'color',min(clrmap(i,:)+0.5,1));
	plot(tIn(idDisp(i,:)), disp(i).Mu(1,:), '-','linewidth',2,'color',min(clrmap(i,:)+0.3,1));
end
plot(tIn, DataOut(1,:), '-','linewidth',3,'color',[0 .7 0]);
%axis([min(tIn) max(tIn) min(Data(1,:))-1 max(Data(1,:))+1]);
ylabel('$y_{t,1}$','fontsize',16,'interpreter','latex');

%Timeline plot of the basis functions activation
axes('Position',[.25 .12 .7 .4]); hold on; 
for i=1:model.nbStates
	patch([tIn(1), tIn, tIn(end)], [0, H(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor',min(clrmap(i,:)+0.2,1),'linewidth',2,'facealpha',.4, 'edgealpha',.4);
end
axis([min(tIn) max(tIn) 0 1.1]);
xlabel('$t$','fontsize',16,'interpreter','latex'); 
ylabel('$h_i(x_t)$','fontsize',16,'interpreter','latex');

%print('-dpng','-r300','graphs/demo_GMRpolyFit01.png');
pause;
close all;