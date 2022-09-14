function [qMatrix]=QuatToMatrix(q)
% Computes a 4x4 matrix from a quaternion, that can be used to implement
% quaternion product.
% João Silvério, 2014

qMatrix = [q.s    -q.v(1) -q.v(2) -q.v(3);
	q.v(1) q.s     -q.v(3) q.v(2);
	q.v(2) q.v(3)  q.s     -q.v(1);
	q.v(3) -q.v(2) q.v(1)  q.s   ];
