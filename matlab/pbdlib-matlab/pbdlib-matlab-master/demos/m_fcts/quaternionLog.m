function a = quaternionLog(q)
% Implements the logarithmic map, which converts a quaternion to axis-angle
% representation.
% João Silvério, Sylvain Calinon, 2015

%Check that a valid quaternion is given as argument
if q.norm < 1E-3 
	a = zeros(3,1);
	return;
end

% %(Re)-normalize quaternion
% q = q/q.norm;

if norm(q.v)<realmin
	a = [0 0 0];
else
	a = 2 * acos(q.s) * q.v / norm(q.v);
end