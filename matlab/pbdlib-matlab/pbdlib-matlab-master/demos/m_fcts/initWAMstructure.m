function [rob,link] = initWAMstructure(robotname)
% WAM kinematic and dynamic model. Parameters from config/wam7w.conf
% João Silvério, 2017

if nargin==0
	robotname = 'WAM';
end

%Kinematic model
alpha=[-pi/2 pi/2 -pi/2 pi/2 -pi/2 pi/2 0];
a = [0 0 0.045 -0.045 0 0 0];
d = [0 0 0.55 0 0.3 0 0.0609]; % tool-free end-effector
% d = [0 0 0.55 0 0.3 0 0.153];

L(1)=Link([0 d(1) a(1) alpha(1) 0 ],'standard');
L(2)=Link([0 d(2) a(2) alpha(2) 0 ],'standard');
L(3)=Link([0 d(3) a(3) alpha(3) 0 ],'standard');
L(4)=Link([0 d(4) a(4) alpha(4) 0 ],'standard');
L(5)=Link([0 d(5) a(5) alpha(5) 0 ],'standard');
L(6)=Link([0 d(6) a(6) alpha(6) 0 ],'standard');
L(7)=Link([0 d(7) a(7) alpha(7) 0 ],'standard');

%Joint angles limits
L(1).qlim = [-2.6 2.6];
L(2).qlim = [-2 2];
L(3).qlim = [-2.8 2.8];
L(4).qlim = [-0.9 3.1];
L(5).qlim = [-4.76 1.24];
L(6).qlim = [-1.6 1.6];
L(7).qlim = [-3 3];

%Dynamic model
%Mass
L(1).m = 10.7677;
L(2).m = 3.8749;
L(3).m = 1.8023;
L(4).m = 2.17266212;
L(5).m = 0.35655692;
L(6).m = 0.40915886;
L(7).m = 0.07548270;

%Center of Gravity
L(1).r = [   -4.43E-3,   121.89E-3,    -0.66E-3 ];
L(2).r = [   -2.37E-3,    31.06E-3,    15.42E-3 ];
L(3).r = [  -38.26E-3,   207.51E-3,     0.03E-3 ];
L(4).r = [ 0.00553408,  0.00006822,  0.11927695 ];
L(5).r = [ 0.00005483,  0.02886286,  0.00148493 ];
L(6).r = [-0.00005923, -0.01686123,  0.02419052 ];
L(7).r = [ 0.00014836,  0.00007252, -0.00335185 ];

%Inertia
L(1).I = [134880.33E-6 -2130.41E-6 -124.85E-6 113283.69E-6 685.55E-6 90463.30E-6 ];
L(2).I = [21409.58E-6    271.72E-6   24.61E-6 13778.75E-6 -1819.20E-6 15589.06E-6];
L(3).I = [59110.77E-6 -2496.12E-6   7.38E-6  3245.50E-6 -17.67E-6 59270.43E-6 ];
L(4).I = [0.01067491 0.00004503 -0.00135557 0.01058659 -0.00011002 0.00282036 ];
L(5).I = [0.00037112 -0.00000008 -0.00000003 0.00019434 -0.00001613  0.00038209 ];
L(6).I = [0.00054889  0.00000019 -0.00000010 0.00023846 -0.00004430 0.00045133 ];
L(7).I = [0.00003911 0.00000019 0.00000000 0.00003877 0.00000000 0.00007614 ];

for i=1:7
	L(i).Jm = 0; %motor inertia
end

%Create robot
rob = SerialLink(L,'name',robotname,'comment','7DOF');
for i=1:length(L)
% 	link(i) = SerialLink(L(1,1:i),'base',T0_r);
    link(i) = SerialLink(L(1,1:i));
end
