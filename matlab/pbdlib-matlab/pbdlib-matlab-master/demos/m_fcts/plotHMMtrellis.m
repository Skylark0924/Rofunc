function h = plotHMMtrellis(Trans, StatesPriors, gridTrans, gridNode, gridInit, colTint)
% Plot the HMM with trellis representation
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Rozo16Frontiers,
%   author="Rozo, L. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
%   title="Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration",
%   journal="Frontiers in Robotics and {AI}",
%   year="2016",
%   month="June",
%   volume="3",
%   number="30",
%   pages="1--11",
%   doi="10.3389/frobt.2016.00030"
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

if nargin<6
	colTint = [1 0 0];
end

[nbStates,nbData] = size(gridNode);
valAlpha = 1;
graphRadius = 1.4;
nodeRadius = .2;
nodePts = 40;
for t=1:nbData
	for i=1:nbStates
		nodePos(:,i,t) = [t; -i*0.5] * graphRadius;
	end
end
h = [];

%Plot initial state
for i=1:nbStates
	dirTmp = nodePos(:,i,2) - nodePos(:,i,1);
	dirTmp = (norm(dirTmp)-2*nodeRadius) * dirTmp/norm(dirTmp);
	offTmp = [dirTmp(2); -dirTmp(1)] / norm(dirTmp);
	posTmp = [0; nodePos(2,i,1)] + nodeRadius * dirTmp/norm(dirTmp) + offTmp*0;
	if gridInit(i)==0
		h = [h plot2DArrow(posTmp, dirTmp, [.8 .8 .8])]; 
	else
		cTmp = colTint(gridInit(i),:);
		h = [h plot2DArrow(posTmp, dirTmp, cTmp,4)]; 
	end
end

%Plot Trans
for t=1:nbData-1
	for i=1:nbStates
		for j=1:nbStates
			dirTmp = nodePos(:,j,t+1) - nodePos(:,i,t);
			dirTmp = (norm(dirTmp)-2*nodeRadius) * dirTmp/norm(dirTmp);
			offTmp = [dirTmp(2); -dirTmp(1)] / norm(dirTmp);
			posTmp = nodePos(:,i,t) + nodeRadius * dirTmp/norm(dirTmp) + offTmp*0;
			if gridTrans(i,j,t)==0
				h = [h plot2DArrow(posTmp,dirTmp,[.8 .8 .8])]; 
			else
				cTmp = colTint(gridTrans(i,j,t),:);
				h = [h plot2DArrow(posTmp,dirTmp,cTmp,4)];
			end
		end
	end
end 

%Plot nodes
for t=1:nbData
	for i=1:nbStates
		a = linspace(0,2*pi,nodePts);
		meshTmp = [cos(a); sin(a)] * nodeRadius + repmat(nodePos(:,i,t),1,nodePts);
		if gridNode(i,t)==0
			h = [h patch(meshTmp(1,:), meshTmp(2,:), [.8 .8 .8],'edgecolor',[.6 .6 .6],'linewidth',2,'facealpha',valAlpha,'edgealpha',valAlpha)];
		else
			cTmp = colTint(gridNode(i,t),:);
			h = [h patch(meshTmp(1,:), meshTmp(2,:), min(cTmp+.5,1),'edgecolor',cTmp,'linewidth',2,'facealpha',valAlpha,'edgealpha',valAlpha)];
		end
		h = [h text(nodePos(1,i,t),nodePos(2,i,t),['$s_' num2str(i) '$'],'interpreter','latex','fontsize',22, ...
			'HorizontalAlignment','center','VerticalAlignment','middle')]; %,'FontWeight','bold'
	end
	h = [h text(nodePos(1,end,t),nodePos(2,end,t)-nodeRadius*2,['$t\!=\!' num2str(t) '$'],'interpreter','latex','fontsize',22, ...
		'HorizontalAlignment','center','VerticalAlignment','middle')];
end

axis equal; axis([nodePos(1,1,1)-nodeRadius*2.4 nodePos(1,1,end)+nodeRadius*1.2 ...
	nodePos(2,end,1)-nodeRadius*2.4 nodePos(2,1,1)+nodeRadius*1.2]);

