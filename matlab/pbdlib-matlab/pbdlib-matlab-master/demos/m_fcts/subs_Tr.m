function T = subs_Tr(theta)

t1=theta(1); t2=theta(2); t3=theta(3); t4=theta(4); t5=theta(5); 
T(1,1) = 0.;
T(1,2) = -15.*sin(t1)+7.50000.*cos(t1);
T(1,3) = -15.*sin(t1)+7.50000.*cos(t1)+15.*sin(t1).*cos(t2)+15.*cos(t1).*sin(t2);
T(1,4) = -15.*sin(t1)+7.50000.*cos(t1)+15.*sin(t1).*cos(t2)+15.*cos(t1).*sin(t2)+15.*(sin(t1).*cos(t2)+cos(t1).*sin(t2)).*cos(t3)+15.*(-1.*sin(t1).*sin(t2)+cos(t1).*cos(t2)).*sin(t3);

T(2,1) = 0.;
T(2,2) = 15.*cos(t1)+7.50000.*sin(t1);
T(2,3) = 15.*cos(t1)+7.50000.*sin(t1)-15.*cos(t1).*cos(t2)+15.*sin(t1).*sin(t2);
T(2,4) = 15.*cos(t1)+7.50000.*sin(t1)-15.*cos(t1).*cos(t2)+15.*sin(t1).*sin(t2)+15.*(-1.*cos(t1).*cos(t2)+sin(t1).*sin(t2)).*cos(t3)+15.*(sin(t1).*cos(t2)+cos(t1).*sin(t2)).*sin(t3);

