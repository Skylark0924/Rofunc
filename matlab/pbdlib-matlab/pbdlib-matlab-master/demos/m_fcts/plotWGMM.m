function h = plotWGMM(Mu, Sigma, color)
%1D plot of a wrapped GMM in polar coordinates
%Sylvain Calinon, 2015

nbStates = size(Mu,2);
nbDrawingSeg = 40;
t = linspace(-pi,pi,nbDrawingSeg);
lightcolor = min(color+0.5,1);
h=[];
for i=1:nbStates
	%Plot covariance
	R = sqrtm(1.0.*Sigma(1:2,1:2,i) + 1E-6*eye(2));
% 	[V,D] = eig(Sigma(1:2,1:2,i) + 0E-6*eye(2));
% 	R = V * D.^.5;	
	
	X = real(R) * [cos(t); sin(t)] + repmat(Mu(1:2,i),1,nbDrawingSeg);
	[Y(1,:), Y(2,:)] = pol2cart(X(1,:), X(2,:));
	h = [h patch(Y(1,:), Y(2,:), lightcolor, 'lineWidth', 1, 'EdgeColor', color)];
	%Plot center
	[Z(1,:), Z(2,:)] = pol2cart(Mu(1,i), Mu(2,i));
	h = [h plot(Z(1,:), Z(2,:), '.', 'lineWidth', 2, 'markersize', 6, 'color', color)];
end