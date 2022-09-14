function [Baxter_l, Baxter_r] = initBaxterStructure
% DH parameters taken from https://groups.google.com/a/rethinkrobotics.com/forum/#!topic/brr-users/5X1-6w-Ja1I
% Sylvain Calinon, 2015

% Left Arm
Ll(1) = Link ([0    0.27035  0.069    -pi/2  0    0], 'standard'); % start at joint s0 and move to joint s1
Ll(2) = Link ([0    0        0         pi/2  0    pi/2], 'standard');        % start at joint s1 and move to joint e0
Ll(3) = Link ([0    0.36435  0.0690   -pi/2  0    0], 'standard'); % start at joint e0 and move to joint e1
Ll(4) = Link ([0    0        0         pi/2  0    0], 'standard');           % start at joint e1 and move to joint w0
Ll(5) = Link ([0    0.37429  0.010    -pi/2  0    0], 'standard');  % start at joint w0 and move to joint w1
Ll(6) = Link ([0    0        0         pi/2  0    0], 'standard');           % start at joint w1 and move to joint w2
Ll(7) = Link ([0    0.229525 0         0     0    0], 'standard');         % start at joint w2 and move to end-effector

% Right Arm
Lr(1) = Link ([0    0.27035  0.069    -pi/2  0     0],'standard');   % start at joint s0 and move to joint s1
Lr(2) = Link ([0    0        0         pi/2  0     pi/2], 'standard');        % start at joint s1 and move to joint e0
Lr(3) = Link ([0    0.36435  0.0690   -pi/2  0     0], 'standard'); % start at joint e0 and move to joint e1
Lr(4) = Link ([0    0        0         pi/2  0     0], 'standard');           % start at joint e1 and move to joint w0
Lr(5) = Link ([0    0.37429  0.010    -pi/2  0     0], 'standard');  % start at joint w0 and move to joint w1
Lr(6) = Link ([0    0        0         pi/2  0     0], 'standard');           % start at joint w1 and move to joint w2
Lr(7) = Link ([0    0.229525 0         0     0     0], 'standard');         % start at joint w2 and move to end-effector

%Mass
Ll(1).m = 5.700440;
Ll(2).m = 3.226980;
Ll(3).m = 4.312720;
Ll(4).m = 2.072060;
Ll(5).m = 2.246650;
Ll(6).m = 1.609790;
Ll(7).m = 0.350930 + 0.191250;

%Center of Gravity
Ll(1).r = [ -0.0511700000000000,     0.0790800000000000,       0.000859999999999956];
Ll(2).r = [  0.00269000000000000,  -0.00529000000000003,     0.0684499999999999];
Ll(3).r = [ -0.0717600000000000,     0.0814900000000001,       0.00131999999999994];
Ll(4).r = [  0.00159000000000006,  -0.0111700000000000,       0.0261799999999999];
Ll(5).r = [ -0.0116799999999999,     0.131110000000000,         0.00459999999999992];
Ll(6).r = [  0.00697000000000011,   0.00599999999999981,     0.0604800000000000];
Ll(7).r = [  0.00513704655280540,   0.000957223615773138,  -0.0668234671142425];

%Inertia
Ll(1).I = [0.04709102262	 -0.00614870039     0.00012787556	0.0359598847	   -0.00078086899	  0.03766976455];
Ll(2).I = [0.0278859752	         -0.00018821993    -0.000300963979	0.0207874929	    0.00207675762     0.01175209419];
Ll(3).I = [0.02661733557	 -0.00392189887     0.00029270634	0.0124800832	   -0.0010838933	  0.02844355207];
Ll(4).I = [0.01318227876	 -0.00019663418     0.00036036173	0.0092685206	    0.0007459496	  0.00711582686];
Ll(5).I = [0.01667742825	 -0.00018657629     0.00018403705	0.0037463115	    0.00064732352	  0.01675457264];
Ll(6).I = [0.00700537914	  0.00015348067    -0.00044384784	0.0055275524	   -0.00021115038	  0.00387607152];
Ll(7).I = [0.00081621358	  0.00012844010     0.000189698911     0.00087350127   0.00010577265	  0.00054941487];

for i=1:7
	Ll(i).Jm = 0; %motor inertia
	Lr(i).Jm = Ll(i).Jm;
	Lr(i).m = Ll(i).m;
	Lr(i).r = Ll(i).r;
	Lr(i).I = Ll(i).I;
end

% Create the Robots Baxter_l (left arm) and Baxter_r (right arm)
Baxter_l = SerialLink(Ll, 'name', 'Baxter_l', 'base' , ...
                      transl(0.024645, 0.219645, 0.118588) * trotz(pi/4)...
                      * transl(0.055695, 0, 0.011038));
Baxter_r = SerialLink(Lr, 'name', 'Baxter_r', 'base' , ...
                      transl(0.024645, -0.219645, 0.118588) * trotz(-pi/4)...
                      * transl(0.055695, 0, 0.011038));
										
Baxter_l.gravity = [0 0 9.81];
Baxter_r.gravity = [0 0 9.81];
