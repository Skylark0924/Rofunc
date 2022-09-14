function h = plotQuatCov(q,Sigma)
% Ioannis Havoutis, Sylvain Calinon, 2016
%
% This function plots a rotation as a 3D frame of reference using rgb 
% colors for each axis, and elipses at the tip of each axis representing 
% the variance.
%	q:		Unit quaternion representing rotation (Quaternion class from rvctools)
%   Sigma:	Covariance matrix (tangent space 3x3)

if (~ishold)
	hold on;
	axis equal;
end

colMat = eye(3);

rotAxis = q.R;
orgPt = [0,0,0];

axisScale = 1;

h(1) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,1), rotAxis(2,1), rotAxis(3,1), ...
	axisScale, 'linewidth', 2, 'color', colMat(:,1));
h(2) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,2), rotAxis(2,2), rotAxis(3,2), ...
	axisScale, 'linewidth', 2, 'color', colMat(:,2));
h(3) = quiver3( orgPt(1), orgPt(2), orgPt(3), rotAxis(1,3), rotAxis(2,3), rotAxis(3,3), ...
	axisScale, 'linewidth', 2, 'color', colMat(:,3));

ar = 1.5;
axis([-ar,ar,-ar,ar,-ar,ar]);
grid on;

view(3);
xlabel('x');ylabel('y');zlabel('z');

% Plot the elipses
colMat = 0.6*ones(3,3);

nbDrawingSeg = 30;
%color = [0,0,1];
valAlpha = 0.5;
%darkcolor = color*0.5; %max(color-0.5,0);
darkcolMat = colMat * 0.5;
t = linspace(-pi, pi, nbDrawingSeg);

for thisAxis = 1 : 3
	Saxis = Sigma;
% 	Saxis(:,thisAxis) = 0;
% 	Saxis(thisAxis,:) = 0;
	
	R = real(sqrtm(1.0.*Saxis));
	e0 = [];
	switch thisAxis
		case 1
			e0 = rotx(pi/2) * R * [zeros(1,nbDrawingSeg); cos(t); sin(t)];
		case 2
			e0 = roty(pi/2) * R * [cos(t); zeros(1,nbDrawingSeg); sin(t)];
		case 3
			e0 = rotz(pi/2) * R * [cos(t); sin(t); zeros(1,nbDrawingSeg)];
	end
	X = rotAxis * e0 + repmat(rotAxis(:,thisAxis), 1, nbDrawingSeg);
	
	h = [h patch(X(1,:), X(2,:), X(3,:), colMat(thisAxis,:), 'lineWidth', 1, 'EdgeColor', darkcolMat(thisAxis,:), 'facealpha', valAlpha,'edgealpha', valAlpha)];
	h = [h plot3(rotAxis(1,thisAxis), rotAxis(2,thisAxis), rotAxis(3,thisAxis), '.', 'markersize', 6, 'color', darkcolMat(thisAxis,:))];
end