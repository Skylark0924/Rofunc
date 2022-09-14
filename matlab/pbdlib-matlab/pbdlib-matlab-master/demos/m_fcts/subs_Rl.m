function R = subs_Rl(theta)

t1=theta(1); t2=theta(2); t3=theta(3); t4=theta(4); t5=theta(5); 
R(1,1,1) = -1.*sin(t1);
R(1,1,2) = sin(t1).*cos(t4)-1.*cos(t1).*sin(t4);
R(1,1,3) = (sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*cos(t5)-1.*(sin(t1).*sin(t4)+cos(t1).*cos(t4)).*sin(t5);

R(1,2,1) = -1.*cos(t1);
R(1,2,2) = sin(t1).*sin(t4)+cos(t1).*cos(t4);
R(1,2,3) = (sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*sin(t5)+(sin(t1).*sin(t4)+cos(t1).*cos(t4)).*cos(t5);

R(2,1,1) = cos(t1);
R(2,1,2) = -1.*cos(t1).*cos(t4)-1.*sin(t1).*sin(t4);
R(2,1,3) = (-1.*cos(t1).*cos(t4)-1.*sin(t1).*sin(t4)).*cos(t5)-1.*(sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*sin(t5);

R(2,2,1) = -1.*sin(t1);
R(2,2,2) = sin(t1).*cos(t4)-1.*cos(t1).*sin(t4);
R(2,2,3) = (-1.*cos(t1).*cos(t4)-1.*sin(t1).*sin(t4)).*sin(t5)+(sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*cos(t5);

