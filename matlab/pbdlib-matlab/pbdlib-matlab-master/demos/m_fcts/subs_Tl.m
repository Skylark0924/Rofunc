function T = subs_Tl(theta)

t1=theta(1); t2=theta(2); t3=theta(3); t4=theta(4); t5=theta(5); 
T(1,1) = 0.;
T(1,2) = -15.*sin(t1)-7.50000.*cos(t1);
T(1,3) = -15.*sin(t1)-7.50000.*cos(t1)+15.*sin(t1).*cos(t4)-15.*cos(t1).*sin(t4);
T(1,4) = -15.*sin(t1)-7.50000.*cos(t1)+15.*sin(t1).*cos(t4)-15.*cos(t1).*sin(t4)+15.*(sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*cos(t5)-15.*(sin(t1).*sin(t4)+cos(t1).*cos(t4)).*sin(t5);

T(2,1) = 0.;
T(2,2) = 15.*cos(t1)-7.50000.*sin(t1);
T(2,3) = 15.*cos(t1)-7.50000.*sin(t1)-15.*cos(t1).*cos(t4)-15.*sin(t1).*sin(t4);
T(2,4) = 15.*cos(t1)-7.50000.*sin(t1)-15.*cos(t1).*cos(t4)-15.*sin(t1).*sin(t4)+15.*(-1.*cos(t1).*cos(t4)-1.*sin(t1).*sin(t4)).*cos(t5)-15.*(sin(t1).*cos(t4)-1.*cos(t1).*sin(t4)).*sin(t5);

