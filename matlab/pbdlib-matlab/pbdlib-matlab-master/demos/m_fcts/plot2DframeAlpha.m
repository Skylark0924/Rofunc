function h = plot2DframeAlpha(rotAxis, orgPt, colMat, scaler, alpha)
% Leonel Rozo, Sylvain Calinon, Andras Kupcsik, 2017
%
% This function plots a 2D frame of reference using rgb colors for each axis.
%   rotAxis:    The rotation matrix representing the 3D frame
%   orgPt:      The origin of the frame

if nargin<3
	colMat = eye(3);
    scaler = 1;
    alpha = 1;
elseif nargin < 4
    scaler = 1;
    alpha = 1;
elseif nargin < 5
    alpha = 1;
end

if (~ishold)
	hold on;
	axis equal;
end

% h(1) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,1), rotAxis(2,1), rotAxis(3,1), 0.2, 'linewidth', 2, 'color', colMat(:,1));
% h(2) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,2), rotAxis(2,2), rotAxis(3,2), 0.2, 'linewidth', 2, 'color', colMat(:,2));
% h(3) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,3), rotAxis(2,3), rotAxis(3,3), 0.2, 'linewidth', 2, 'color', colMat(:,3));

%for faster plot
% h(1) = plot([orgPt(1) orgPt(1)+rotAxis(1,1) * scaler], [orgPt(2) orgPt(2)+rotAxis(2,1) * scaler], 'linewidth', 2, 'color', colMat(:,1));
% h(2) = plot([orgPt(1) orgPt(1)+rotAxis(1,2) * scaler], [orgPt(2) orgPt(2)+rotAxis(2,2) * scaler], 'linewidth', 2, 'color', colMat(:,2));

x =  [orgPt(1)  orgPt(1)+rotAxis(1,1) * scaler]';
y = [orgPt(2) orgPt(2)+rotAxis(2,1) * scaler]';

h(1) = patchline(x, y, ...
    'EdgeColor','red','LineWidth',2, 'edgealpha', alpha);

 x =  [orgPt(1) orgPt(1)+rotAxis(1,2) * scaler]';
y = [orgPt(2) orgPt(2)+rotAxis(2,2) * scaler]';

h(2) = patchline(x, y, ...
    'EdgeColor','green','LineWidth',2, 'edgealpha', alpha);