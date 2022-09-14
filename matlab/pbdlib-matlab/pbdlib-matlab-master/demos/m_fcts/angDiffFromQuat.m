function omega = angDiffFromQuat(Q1, Q2)

q1 = Quaternion(Q1);
q2 = Quaternion(Q2);
	
%Take closest difference
if(q1.inner(q2)<0)
	q2 = Quaternion(-q2.double);
end
		
omega = 2 * quaternionLog(q1 * q2.inv)';