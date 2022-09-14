function diff = quaternionDiff(q1, q2)
% Computes the difference between two quaternions
q1 = Quaternion(q1);
q2 = Quaternion(q2);

if (q1.inner(q2) < 0)
	q2 = Quaternion(-q2.double);
	disp('Inverting quaternion');
end

diff = quaternionLog(q1*q2.inv)*2.0;
