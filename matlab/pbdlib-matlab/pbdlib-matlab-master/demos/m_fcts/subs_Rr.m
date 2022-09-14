function R = subs_Rr(theta)

t1=theta(1); t2=theta(2); t3=theta(3); t4=theta(4); t5=theta(5); 
R(1,1,1) = -1.*sin(t1);
R(1,1,2) = sin(t1).*cos(t2)+cos(t1).*sin(t2);
R(1,1,3) = (sin(t1).*cos(t2)+cos(t1).*sin(t2)).*cos(t3)+(-1.*sin(t1).*sin(t2)+cos(t1).*cos(t2)).*sin(t3);

R(1,2,1) = -1.*cos(t1);
R(1,2,2) = -1.*sin(t1).*sin(t2)+cos(t1).*cos(t2);
R(1,2,3) = -1.*(sin(t1).*cos(t2)+cos(t1).*sin(t2)).*sin(t3)+(-1.*sin(t1).*sin(t2)+cos(t1).*cos(t2)).*cos(t3);

R(2,1,1) = cos(t1);
R(2,1,2) = -1.*cos(t1).*cos(t2)+sin(t1).*sin(t2);
R(2,1,3) = (-1.*cos(t1).*cos(t2)+sin(t1).*sin(t2)).*cos(t3)+(sin(t1).*cos(t2)+cos(t1).*sin(t2)).*sin(t3);

R(2,2,1) = -1.*sin(t1);
R(2,2,2) = sin(t1).*cos(t2)+cos(t1).*sin(t2);
R(2,2,3) = -1.*(-1.*cos(t1).*cos(t2)+sin(t1).*sin(t2)).*sin(t3)+(sin(t1).*cos(t2)+cos(t1).*sin(t2)).*cos(t3);

