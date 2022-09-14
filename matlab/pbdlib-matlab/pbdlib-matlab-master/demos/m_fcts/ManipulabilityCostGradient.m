function Cgrad = ManipulabilityCostGradient(typeCost, Me_c, Me_d, q)

% Leonel Rozo, 2017
%
% This function computes a symbolic equation representing the gradient of a
% cost function that evaluates how similar two manipulability ellipsoids
% are.
%
% Parameters:
%   - typeCost: Type of cost function to be used
%   - Me_c:     Manipulability ellipsoid at current time step (symbolic)
%   - Me_d:     Desired manipulability ellipsoid
%   - q:        Vector of symbolic variables representing the robot joints
% 
% Returns:
%   - Cgrad:    Symbolic equation of the cost gradient


%% Computation of cost function and its gradient
% Computation of the cost function
[eVc, ~] = eig(Me_d);
des_vec = eVc(:,1)/norm(eVc(:,1)); % Desired VME major axis

switch typeCost
	case 1 % A. Ajoudani approach (Major-axis alignment)
		% Minus sign is included here because the cost function is being
		% minimized in the nullspace
    C = -inv((des_vec)' * Me_c * (des_vec));
    
	case 2 % Squared Stein divergence 
		C = log(det(0.5*(Me_d+Me_c))) - 0.5*log(det(Me_d*Me_c));
end

% Symbolic cost gradient computation
C_gradient = [];
for i = 1 : length(q)
	C_gradient = [C_gradient ; diff(C,q(i))];
end

% Creating a MATLAB function from symbolic variable
Cgrad =  matlabFunction(C_gradient);


