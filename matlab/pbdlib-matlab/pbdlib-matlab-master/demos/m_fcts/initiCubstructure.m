function [rob,link] = initiCubstructure(robotname)
%Initialization of the kinematic chain of iCub's right arm from DH params (from trunk to hand palm)
%First run 'startup_rvc' from the robotics toolbox
%Source: http://wiki.icub.org/wiki/ICubFowardKinematics_right
%João Silvério, Sylvain Calinon, 2015

if nargin==0
	robotname = 'iCub right arm';
end

% D-H parameters
alpha=[pi/2 pi/2 pi/2 pi/2 -pi/2 -pi/2 pi/2 pi/2 pi/2 0];
a = 0.001*[32  0     -23.3647  0      0 -15.0    15.0   0   0  62.5];
d = 0.001*[0   -5.5  -143.3   -107.74 0 -152.28  0    -141.3 0  25.98]; % tool-free end-effector
offset = [0 -pi/2 -15*pi/180-pi/2 -pi/2 -pi/2 -15*pi/180-pi/2 0 -pi/2 pi/2 pi]; %theta offset
 
L(1)=Link([0 d(1) a(1) alpha(1) 0 offset(1)],'standard');
L(2)=Link([0 d(2) a(2) alpha(2) 0 offset(2)],'standard');
L(3)=Link([0 d(3) a(3) alpha(3) 0 offset(3)],'standard');
L(4)=Link([0 d(4) a(4) alpha(4) 0 offset(4)],'standard');
L(5)=Link([0 d(5) a(5) alpha(5) 0 offset(5)],'standard');
L(6)=Link([0 d(6) a(6) alpha(6) 0 offset(6)],'standard');
L(7)=Link([0 d(7) a(7) alpha(7) 0 offset(7)],'standard');
L(8)=Link([0 d(8) a(8) alpha(8) 0 offset(8)],'standard');
L(9) =Link([0 d(9) a(9) alpha(9) 0 offset(9)],'standard');
L(10)=Link([0 d(10) a(10) alpha(10) 0 offset(10)],'standard');

%Joint angles limits
L(1).qlim = (offset(1) + (pi/180)*[-22 84]);
L(2).qlim = (offset(2) + (pi/180)*[-39 39]);
L(3).qlim = (offset(3) + (pi/180)*[-59 59]);
L(4).qlim = (offset(4) + (pi/180)*[5 -95]);
L(5).qlim = (offset(5) + (pi/180)*[0 160.8]);
L(6).qlim = (offset(6) + (pi/180)*[-37 100]);
L(7).qlim = (offset(7) + (pi/180)*[5.5 106]);
L(8).qlim = (offset(8) + (pi/180)*[-50 50]);
L(9).qlim = (offset(9) + (pi/180)*[10 -65]);
L(10).qlim = (offset(10) + (pi/180)*[-25 25]);

% Mapping between root frame and link 0 (hom. transformation matrix)
T0_r = [0 -1 0 0; 0 0 -1 0; 1 0 0 0; 0 0 0 1];

%Create robot
rob = SerialLink(L,'name',robotname,'base',T0_r,'comment','10DOF');
for i=1:length(L)
	link(i) = SerialLink(L(1,1:i),'base',T0_r);
end
