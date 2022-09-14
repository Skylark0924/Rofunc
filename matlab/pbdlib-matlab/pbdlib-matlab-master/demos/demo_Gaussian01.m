function demo_Gaussian01
% Use of Chi-square values to determine the percentage of data within the contour of a multivariate normal distribution.
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
% 	author="Calinon, S.",
% 	title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
% 	booktitle="Mixture Models and Applications",
% 	publisher="Springer",
% 	editor="Bouguila, N. and Fan, W.", 
% 	year="2019",
% 	pages="39--57",
% 	doi="10.1007/978-3-030-23876-6_3"
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
model.nbVar = 2; %Number of variables [x1,x2]
nbStd = 3; %Number of standard deviations to consider

%For the normal distribution with D=1, the values less than one standard deviation away from the mean account for 68.27% of the set; 
%while two standard deviations from the mean account for 95.45%; and three standard deviations account for 99.73%.
%For D>1:
%https://en.wikipedia.org/wiki/Chi-squared_distribution
%http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
%ChiSq = [4.605 5.991 9.210]; %90% 95% 99% intervals for D=2
ChiSq = [2.41 	3.22 	4.60]; %70% 80% 90% intervals for D=2


% %% Generate random normally distributed data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbData = 1000;
% Data = [];
% r = rand(model.nbVar,10)-0.5;
% [V,D] = eigs(cov(r'));
% %D(model.nbFA+1:end,model.nbFA+1:end) = 0;
% R = real(V*D.^.5);
% b = (rand(model.nbVar,1)-0.5) * 2;
% Data = R * randn(model.nbVar,nbData) + repmat(b,1,nbData); % + randn(model.nbVar,nbData)*1E-3;


%% Load  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/faithful.mat');
Data = faithful';
nbData = size(Data,2);


% %% Load handwriting data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbData = 100;
% nbSamples = 10;
% demos=[];
% load('data/2Dletters/P.mat');
% %nbSamples = length(demos);
% Data=[];
% for n=1:nbSamples
% 	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
% 	Data = [Data s(n).Data]; 
% end
% nbData = size(Data,2); %Total number of datapoints


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Mu = mean(Data,2);
model.Sigma = cov(Data');
[V,D] = eigs(model.Sigma);

%Compute threshold
for n=1:nbStd
	thrStd(n) = gaussPDF(model.Mu+V(:,1)*(ChiSq(n)*D(1,1))^.5, model.Mu, model.Sigma);
end
%Estimate number of points within n standard deviations
H = gaussPDF(Data, model.Mu, model.Sigma);
for n=1:nbStd
	id(:,n) = H>=thrStd(n);
	ratio(n) = sum(id(:,n)) / nbData;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 10 4],'position',[10,10,1300,500]); 
clrmap = min(lines(nbStd)+0.6,1);
mrg = (max(Data') - min(Data')) * 0.52;
for nn=1:nbStd
	subplot(1,nbStd,nn); hold on; axis off; title(['Contour of ' num2str(ChiSq(nn),'%.2f') ' \Sigma'],'fontsize',16);
	for n=nn:nn %-1:1
		plotGMM(model.Mu, model.Sigma*ChiSq(n), clrmap(n,:),1);
	end
	plot(Data(1,1:200), Data(2,1:200), 'o','markersize',4,'color',[.5 .5 .5]);
	for n=nn:nn %-1:1
		hg = plot(Data(1,id(1:200,n)), Data(2,id(1:200,n)), '.','markersize',16,'color',max(clrmap(n,:)-0.2,0));
	end
	hl = legend(hg, sprintf('%s\n%s',[num2str(sum(id(:,nn))) ' datapoints'], ['(' num2str(ratio(nn)*100,'%.0f') '% of total number)']));
	set(hl,'Box','off','Location','SouthOutside','fontsize',16);
	axis([min(Data(1,:))-mrg(1) max(Data(1,:))+mrg(1) min(Data(2,:))-mrg(2) max(Data(2,:))+mrg(2)]);
	%axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
end

%print('-dpng','graphs/demo_Gaussian01.png');
pause;
close all;