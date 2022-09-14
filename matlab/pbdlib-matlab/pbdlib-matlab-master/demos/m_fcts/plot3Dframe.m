function h = plot3Dframe(rotAxis, orgPt, colMat, lineWidth)
% Leonel Rozo, Sylvain Calinon, 2015
%
% This function plots a 3D frame of reference using rgb colors for each axis.
%   rotAxis:    The rotation matrix representing the 3D frame
%   orgPt:      The origin of the frame

if nargin<1
	rotAxis = eye(3);
end

if nargin<2
	orgPt = zeros(3,1);
end
	
if nargin<3
	colMat = eye(3);
	lineWidth = 2;
end

if nargin < 4
	lineWidth = 2;
end

if (~ishold)
	hold on;
	axis equal;
end

% h(1) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,1), rotAxis(2,1), rotAxis(3,1), 0.2, 'linewidth', 2, 'color', colMat(:,1));
% h(2) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,2), rotAxis(2,2), rotAxis(3,2), 0.2, 'linewidth', 2, 'color', colMat(:,2));
% h(3) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,3), rotAxis(2,3), rotAxis(3,3), 0.2, 'linewidth', 2, 'color', colMat(:,3));

%for faster plot
h(1) = plot3([orgPt(1) orgPt(1)+rotAxis(1,1)], [orgPt(2) orgPt(2)+rotAxis(2,1)], [orgPt(3) orgPt(3)+rotAxis(3,1)], 'linewidth', lineWidth, 'color', colMat(:,1));
h(2) = plot3([orgPt(1) orgPt(1)+rotAxis(1,2)], [orgPt(2) orgPt(2)+rotAxis(2,2)], [orgPt(3) orgPt(3)+rotAxis(3,2)], 'linewidth', lineWidth, 'color', colMat(:,2));
h(3) = plot3([orgPt(1) orgPt(1)+rotAxis(1,3)], [orgPt(2) orgPt(2)+rotAxis(2,3)], [orgPt(3) orgPt(3)+rotAxis(3,3)], 'linewidth', lineWidth, 'color', colMat(:,3));
