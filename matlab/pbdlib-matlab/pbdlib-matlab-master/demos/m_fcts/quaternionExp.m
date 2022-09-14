function q = quaternionExp(a)
% Computes the exponential map of a quaternion (axis-angle to quaternion).
% João Silvério, 2014

a = a(:)';

if norm(a)>pi
	disp([num2str(norm(a)) '>pi']);
end

if all(a==0)
	q = Quaternion([1 0 0 0]); % corrected from the paper
else
	v = cos(norm(a)/2);
	u = sin(norm(a)/2) * a/norm(a);
	q = Quaternion([v u]);
end