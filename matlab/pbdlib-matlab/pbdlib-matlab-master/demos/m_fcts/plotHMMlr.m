function plotHMMlr(Trans, StatesPriors)
% Plot the transition graph of an HMM with left-right structure
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
	
nbStates = size(Trans,1);
valAlpha = 1;
graphRadius = 0.7;
nodeRadius = .2;
nodePts = 40;
for i=1:nbStates
	nodePos(:,i) = [i; 0] * graphRadius;
end
clrmap = lines(nbStates);

for i=1:nbStates
	%Plot StatesPriors 
	if StatesPriors(i)>0.5
		posTmp = [nodePos(1,i)-nodeRadius*2; 0];
		dirTmp = nodePos(:,i) - posTmp;
		dirTmp = (norm(dirTmp)-nodeRadius) * dirTmp/norm(dirTmp);
		plot2DArrow(posTmp, dirTmp, [.8 .8 .8]-StatesPriors(i)*.8);
	end
	
% 	%Plot self-transitions
% 	d = nodeRadius*1.2;
% 	posTmp = [nodePos(1,i); d];
% 	R = nodeRadius;
% 	r = nodeRadius*0.5; 
% 	aTmp = asin((4*d^2*R^2-(d^2-r^2+R^2)^2)^.5/(2*d*r));
% 	a = linspace(pi/2+pi-aTmp, pi/2-pi+aTmp, nodePts);
% 	meshTmp = [cos(a); sin(a)] * r + repmat(posTmp,1,nodePts);
% 	plot(meshTmp(1,:), meshTmp(2,:), 'color',[0 0 0],'linewidth',2); %[.8 .8 .8]-Trans(i,i)*.8
% 	plot2DArrow(meshTmp(:,end-10), meshTmp(:,end)-meshTmp(:,end-10), [0 0 0]); %[.8 .8 .8]-Trans(i,i)*.8

	%Plot Trans i+1 (next state)
	for j=i+1:min(i+1,nbStates)
		dirTmp = nodePos(:,j)-nodePos(:,i);
		dirTmp = (norm(dirTmp)-2*nodeRadius) * dirTmp/norm(dirTmp);
		offTmp = [dirTmp(2); -dirTmp(1)] / norm(dirTmp);
		posTmp = nodePos(:,i) + nodeRadius * dirTmp/norm(dirTmp) + offTmp*0;
		plot2DArrow(posTmp, dirTmp, [0 0 0]); %[.8 .8 .8]-Trans(i,j)*.8
	end
	
% 	%Plot Trans i+2
% 	for j=i+2:min(i+2,nbStates)	
% 		d = nodeRadius*2;
% 		posTmp = [nodePos(1,i+1); -d*.15];
% 		R = nodeRadius*3.1;
% 		r = nodeRadius*3.1; 
% 		aTmp = asin((4*d^2*R^2-(d^2-r^2+R^2)^2)^.5/(2*d*r));
% 		a = linspace(pi/2-aTmp, pi/2+aTmp, nodePts);
% 		meshTmp = -[cos(a); sin(a)*.5] * r + repmat(posTmp,1,nodePts);
% 		plot(meshTmp(1,:), meshTmp(2,:), 'color',[0 0 0],'linewidth',2); %[.8 .8 .8]-Trans(i,i)*.8
% 		plot2DArrow(meshTmp(:,end-5), meshTmp(:,end)-meshTmp(:,end-5), [0 0 0]); %[.8 .8 .8]-Trans(i,i)*.8
% 	end
end

%Plot nodes
for i=1:nbStates
	a = linspace(0,2*pi,nodePts);
	meshTmp = [cos(a); sin(a)] * nodeRadius + repmat(nodePos(:,i),1,nodePts);
	patch(meshTmp(1,:), meshTmp(2,:), clrmap(i,:),'edgecolor',clrmap(i,:)*0.5, 'facealpha', valAlpha,'edgealpha', valAlpha);
	%text(nodePos(1,i),nodePos(2,i),num2str(i),'HorizontalAlignment','center','FontWeight','bold','fontsize',20);
	text(nodePos(1,i)-.03,nodePos(2,i)-.01,['$s_' num2str(i) '$'],'interpreter','latex','fontsize',58, ...
			'HorizontalAlignment','center','VerticalAlignment','middle'); %,'FontWeight','bold'
end

axis equal;
