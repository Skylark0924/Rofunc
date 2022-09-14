function demo_Bezier01
% Bezier curves as a superposition of Bernstein polynomials
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Dimension of datapoint
nbDeg = 3; %Degree of the Bezier curve
nbData = 100; %Number of datapoints in a trajectory

p = rand(nbVar,nbDeg+1); %Control points
tl = linspace(0,1,nbData); %Time range


%% Bezier curve as a superposition of Bernstein polynomials
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = zeros(nbDeg,nbData);
for i=0:nbDeg
	B(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-tl).^(nbDeg-i) .* tl.^i; %Bernstein basis functions (Eq.(12)
end
x = p * B; %Reconstruction of signal


%% Bezier curve computed recursively
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pTmp = zeros(nbDeg, nbDeg, nbVar, nbData);
% for t=1:nbData
% 	for i=1:nbDeg
% 		for j=i:nbDeg
% 			if i==1
% 				pTmp(i,j,:,t) = p(:,j) * (1-tl(t)) + p(:,j+1) * tl(t);
% 			else
% 				pTmp(i,j,:,t) = pTmp(i-1,j-1,:,t) * (1-tl(t)) + pTmp(i-1,j,:,t) * tl(t);
% 			end
% 		end
% 	end
% end
% x(:,:) = pTmp(end,end,:,:);

% Btmp = zeros(nbDeg, nbDeg+1, nbData);
% for t=1:nbData
% 	Btmp(1,1,t) = tl(t);
% 	Btmp(1,2,t) = 1 - tl(t);
% 	for i=2:nbDeg
% 		for j=1:i
% 			[i, j]
% 			Btmp(i,j,t) = Btmp(i-1,j,t) * (1-tl(t)) + Btmp(i-1,j+1,t) * tl(t);
% 		end
% 	end
% end
% B(:,:) = Btmp(end,:,:);
% x = p * B; %Reconstruction of signal


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800]); 
subplot(2,1,1); hold on; axis off;
plot(p(1,:), p(2,:), '.','markersize',30,'color',[.6 .6 .6]);
if nbDeg==3
	plot(p(1,[1:2]), p(2,[1:2]), '-','linewidth',2,'color',[.6 .6 .6]);
	plot(p(1,[3:4]), p(2,[3:4]), '-','linewidth',2,'color',[.6 .6 .6]);
end
plot(x(1,:), x(2,:), '-','linewidth',3,'color',[0 0 0]);
subplot(2,1,2); hold on;
for i=0:nbDeg
	plot(tl, B(i+1,:),'linewidth',3);
end
set(gca,'xtick',[0,1],'ytick',[0,1],'fontsize',16);
xlabel('t','fontsize',18); ylabel('b_i','fontsize',18);

pause;
close all;