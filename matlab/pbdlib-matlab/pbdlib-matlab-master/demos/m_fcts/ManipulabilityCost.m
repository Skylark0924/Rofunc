function Cost = ManipulabilityCost(typeCost, Me_c, Me_d)

% Leonel Rozo, 2017
%
% This function computes a symbolic equation representing cost function 
% that evaluates how similar two manipulability ellipsoids are.
%
% Parameters:
%   - typeCost: Type of cost function to be used
%   - Me_c:     Manipulability ellipsoid at current time step (symbolic)
%   - Me_d:     Desired manipulability ellipsoid
% 
% Returns:
%   - Cost:    Symbolic equation of the cost function


%% Computation of cost function and its gradient
% Computation of the cost function
[eVc, ~] = eig(Me_d);
des_vec = eVc(:,1)/norm(eVc(:,1)); % Desired VME major axis

switch typeCost
	case 1 % A. Ajoudani approach
		% Minus sign is included here because the cost function is being
		% minimized in the nullspace
    C = -inv((des_vec)' * Me_c * (des_vec));
    
	case 2 % Squared Stein divergence 
		C = log(det(0.5*(Me_d+Me_c))) - 0.5*log(det(Me_d*Me_c));
end

% Creating a MATLAB function from symbolic variable
Cost =  matlabFunction(C);


