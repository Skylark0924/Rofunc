function [rob,link] = initCOMAN_Larm(robotname)
%Initialization of the kinematic chain of COMAN's left arm from DH params (from trunk to hand palm)
%First run 'startup_rvc' from the robotics toolbox
%João Silvério 2015

if nargin==0
	robotname = 'COMAN left arm';
end

% Mapping between root frame and link 0 (hom. transformation matrix)
% from urdf2dh: H0 (0.34202 0.0 0.939693 0.020282 0.0 1.0 0.0 0.0 -0.939693 0.0 0.34202 0.119121 0.0 0.0 0.0 1.0)
T0_r = [0.34202 0.0 0.939693 0.020282 ; 0.0 1.0 0.0 0.0 ; -0.939693 0.0 0.34202 0.119121 ; 0.0 0.0 0.0 1.0];

% from urdf2dh: 
% link_0 (A -0.0)      (D 0.0)      (alpha -1.570796) (offset -0.0)      (min -30.00007)   (max 29.994341)
% link_1 (A -0.0)      (D 0.0)      (alpha 1.570796)  (offset -1.22173)  (min -20.001957)  (max 50.002027)
% link_2 (A -0.014977) (D 0.205208) (alpha -1.570796) (offset 0.0)       (min -80.002097)  (max 79.996367)
% link_3 (A -0.0)      (D 0.1558)   (alpha 1.570796)  (offset 1.570796)  (min -195.000456) (max 95.002132)
% link_4 (A -0.0)      (D 0.0)      (alpha 1.570796)  (offset -1.570796) (min -17.999469)  (max 119.977362)
% link_5 (A 0.015)     (D -0.18)    (alpha -1.570796) (offset 1.570796)  (min -90.00021)   (max 90.00021)
% link_6 (A -0.015)    (D 0.0)      (alpha 1.570796)  (offset 0.0)       (min -135.000316) (max 0.0)
% link_7 (A -0.0)      (D -0.19468) (alpha -1.570796) (offset 0.0)       (min -90.00021)   (max 90.00021)
% link_8 (A -0.0)      (D 0.0)      (alpha 1.570796)  (offset 1.570796)  (min -30.022988)  (max 30.022988)
% link_9 (A -0.0)      (D 0.0)      (alpha 1.570796)  (offset -1.570796) (min -44.998673)  (max 79.927612)

% D-H parameters
a = [0.0 0.0 -0.014977 0.0 0.0 0.015 -0.015 0.0 0.0 0.0];
d = [0.0 0.0 0.205208  0.1558 0.0 -0.18 0.0 -0.19468 0.0 0.0]; % tool-free end-effector
alpha=[-pi/2 pi/2 -pi/2 pi/2 pi/2 -pi/2 pi/2 -pi/2 pi/2 pi/2];
offset = [0.0 -1.22173 0.0 pi/2 -pi/2 pi/2 0.0 0.0 pi/2 -pi/2]; %theta offset
jlim_min = (pi/180)*[-30.00007 -20.001957 -80.002097 -195.000456 -17.999469 -90.00021 -135.000316 -90.00021 -30.022988 -44.998673];
jlim_max = (pi/180)*[29.994341 50.002027 79.996367 95.002132 119.977362 90.00021 0.0 90.00021 30.022988 79.927612];

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
L(1).qlim = offset(1) + [jlim_min(1) jlim_max(1)];
L(2).qlim = offset(2) + [jlim_min(2) jlim_max(2)];
L(3).qlim = offset(3) + [jlim_min(3) jlim_max(3)];
L(4).qlim = offset(4) + [jlim_min(4) jlim_max(4)];
L(5).qlim = offset(5) + [jlim_min(5) jlim_max(5)];
L(6).qlim = offset(6) + [jlim_min(6) jlim_max(6)];
L(7).qlim = offset(7) + [jlim_min(7) jlim_max(7)];
L(8).qlim = offset(8) + [jlim_min(8) jlim_max(8)];
L(9).qlim = offset(9) + [jlim_min(9) jlim_max(9)];
L(10).qlim = offset(10) + [jlim_min(10) jlim_max(10)];

% Tool transform
TN = [0.0 -1.0 -0.0 0.0 ; 1.0 0.0 0.0 0.0 ; 0.0 -0.0 1.0 -0.07 ; 0.0 0.0 0.0 1.0];

%Create robot
rob = SerialLink(L,'name',robotname,'base',T0_r,'tool',TN,'comment','10DOF');
for i=1:length(L)
	link(i) = SerialLink(L(1,1:i),'base',T0_r);
end
