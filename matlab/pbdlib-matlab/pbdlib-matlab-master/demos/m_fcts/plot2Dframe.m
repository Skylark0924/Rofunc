function h = plot2Dframe(rotAxis, orgPt, colMat, lw)
% Leonel Rozo, Sylvain Calinon, Andras Kupcsik, 2017
%
% This function plots a 2D frame of reference using rgb colors for each axis.
%   rotAxis:    The (scaled) rotation matrix representing the 3D frame
%   orgPt:      The origin of the frame

if nargin<3
	colMat = eye(3);
end
if nargin < 4
	lw = 2;
end

if (~ishold)
	hold on;
	axis equal;
end

% h(1) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,1), rotAxis(2,1), rotAxis(3,1), 0.2, 'linewidth', 2, 'color', colMat(:,1));
% h(2) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,2), rotAxis(2,2), rotAxis(3,2), 0.2, 'linewidth', 2, 'color', colMat(:,2));
% h(3) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,3), rotAxis(2,3), rotAxis(3,3), 0.2, 'linewidth', 2, 'color', colMat(:,3));

%for faster plot
h = [];
for t=1:size(rotAxis,3)
	h = [h, plot([orgPt(1,t) orgPt(1,t)+rotAxis(1,1,t)], [orgPt(2,t) orgPt(2,t)+rotAxis(2,1,t)], 'linewidth', lw, 'color', colMat(:,1))];
	h = [h, plot([orgPt(1,t) orgPt(1,t)+rotAxis(1,2,t)], [orgPt(2,t) orgPt(2,t)+rotAxis(2,2,t)], 'linewidth', lw, 'color', colMat(:,2))];
	h = [h, plot(orgPt(1,t), orgPt(2,t), '.','markersize',lw*10,'color',[0,0,0])];
end
