function demo_spring01
% Influence of the damping ratio in mass-spring-damper systems
%
% If this code is useful for your research, please cite the related publication:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Number of variables 
kP = 50; %Stiffness gain
dt = 0.01; %Duration of time step
nbData = 200; %Length of each trajectory

xTar = [0;0];

%listKv = linspace((2*kP)^.5, 2*kP^.5, 5); %From ideal damping to critically damped
listKv = [.5.*(2*kP)^.5, (2*kP)^.5, 2*kP^.5, 4*kP^.5]; %underdamped, ideal damping, critically damped, overdamped

clrmap = lines(length(listKv));
f1 = figure('position',[10,10,1300,450]); 

f2 = figure('position',[1350,10,500,450]); 
subplot(2,1,1); hold on;
plot([1,nbData]*dt, [xTar(1) xTar(1)], '-','linewidth',2,'color',[0 0 0]);
subplot(2,1,2); hold on;
plot([1,nbData]*dt, [0 0], '-','linewidth',2,'color',[0 0 0]);


for n=1:length(listKv)
kV = listKv(n); %Damping gain (with ideal underdamped damping ratio)
L = [eye(nbVar)*kP, eye(nbVar)*kV]; %Feedback term

%% Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = [-1;-.2];
dx = zeros(nbVar,1);
for t=1:nbData
	Data(:,t) = [x; dx];
	ddx =  L * [xTar-x; -dx] ;  
	dx = dx + ddx * dt;
	x = x + dx * dt;
end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spatial plot
figure(f1);
subplot(2,2,n); hold on; axis off;
plot(Data(1,:),Data(2,:),'-','linewidth',2,'color',clrmap(n,:)); 
plot(Data(1,1:5:end),Data(2,1:5:end),'.','markersize',10,'color',clrmap(n,:)*.8);
plot(xTar(1),xTar(2),'.','markersize',24,'color',[0 0 0]);
axis([-1,.3,-.2,.1]); axis equal;

%Timeline plots
figure(f2);
%x
subplot(2,1,1); hold on;
plot([1:nbData]*dt, Data(1,:), '-','linewidth',2,'color',clrmap(n,:));
axis([dt nbData*dt -1 .3]);
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','fontsize',18,'interpreter','latex');
ylabel('$x_1$','fontsize',18,'interpreter','latex');
%dx
subplot(2,1,2); hold on;
plot([1:nbData]*dt, Data(3,:), '-','linewidth',2,'color',clrmap(n,:));
axis([dt nbData*dt -1.5 4.6]);
set(gca,'xtick',[],'ytick',[0]);
xlabel('$t$','fontsize',18,'interpreter','latex');
ylabel('$\dot{x}_1$','fontsize',18,'interpreter','latex');

end %n

% figure(f1);
% print('-dpng','graphs/demo_spring01a.png');
% figure(f2);
% print('-dpng','graphs/demo_spring01b.png');

pause;
close all;