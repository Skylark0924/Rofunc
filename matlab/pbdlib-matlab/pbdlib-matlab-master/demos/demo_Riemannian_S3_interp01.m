function demo_Riemannian_S3_interp01
% Interpolation of unit quaternions (orientation) by relying on Riemannian manifold, providing the same result as SLERP interpolation
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Sylvain Calinon, https://calinon.ch/
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 3; %Dimension of the tangent space
nbVarMan = 4; %Dimension of the manifold
nbStates = 2; %Number of states
nbData = 100; %Number of interpolation steps
% nbIter = 20; %Number of iteration for the Gauss Newton algorithm

x = rand(nbVarMan,nbStates) - 0.5;
for i=1:nbStates
	x(:,i) = x(:,i) / norm(x(:,i));
end


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];
xi = zeros(nbVarMan,nbData);
% xtmp = x(:,1);

for t=1:nbData
% 	%Interpolation between more than 2 points can be computed in an iterative form
% 	for n=1:nbIter
% 		utmp = zeros(nbVar,1);
% 		for i=1:nbStates
% 			utmp = utmp + w(i,t) * logmap(x(:,i), xtmp);
% 		end
% 		xtmp = expmap(utmp, xtmp);
% 	end
% 	xi(:,t) = xtmp;

	%Interpolation between two covariances can be computed in closed form
	xi(:,t) = expmap(w(2,t) * logmap(x(:,2),x(:,1)), x(:,1));
end


%% SLERP interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:nbData
	xi2(:,t) = quatinterp(x(:,1)', x(:,2)', w(2,t), 'slerp');
end


% %% Naive interpolation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% R1 = quat2rotm(x(:,1)');
% R2 = quat2rotm(x(:,2)');
% for t=1:nbData
% 	[U,~,V] = svd(w(1,t) .* R1 + w(2,t) .* R2);
% 	R = U * V';
% 	xi2(:,t) = rotm2quat(R);
% end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,900,1250]);
for i=1:nbVarMan
	subplot(nbVarMan,1,i); hold on;
	for n=1:nbStates
		plot([1,nbData],[x(i,n),x(i,n)],'-','linewidth',2,'color',[0 0 0]);
		plot([1,nbData],[-x(i,n),-x(i,n)],'--','linewidth',2,'color',[0 0 0]);
	end
	h(1) = plot(xi(i,:),'-','linewidth',2,'color',[.8 0 0]);
	h(2) = plot(xi2(i,:),':','linewidth',2,'color',[0 .7 0]);
	if i==1
% 		legend(h,'geodesic','SLERP');
		legend(h,'geodesic','Naive interpolation');
	end
	ylabel(['q_' num2str(i)]);
	axis([1, nbData -1 1]);
end
xlabel('t');

%3D plot
figure('position',[920,10,900,1250]); 
subplot(2,1,1); hold on; axis off; rotate3d on;
colormap([.9 .9 .9]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
plot3Dframe(quat2rotm(x(:,end)'), zeros(3,1), eye(3)*.3);
view(3); axis equal; axis tight; axis vis3d;  
h=[];
for t=1:nbData
	if mod(t,10)~=0
		delete(h);
	end
	h = plot3Dframe(quat2rotm(xi(:,t)'), zeros(3,1));
	drawnow;
end

%SLERP interpolation
subplot(2,1,2); hold on; axis off; rotate3d on;
colormap([.9 .9 .9]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
plot3Dframe(quat2rotm(x(:,end)'), zeros(3,1), eye(3)*.3);
view(3); axis equal; axis tight; axis vis3d;  
h=[];
for t=1:nbData
	if mod(t,10)~=0
		delete(h);
	end
	h = plot3Dframe(quat2rotm(xi2(:,t)'), zeros(3,1));
	drawnow;
end

% %Naive interpolation
% subplot(2,1,2); hold on; axis off; rotate3d on;
% colormap([.9 .9 .9]);
% [X,Y,Z] = sphere(20);
% mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
% plot3Dframe(quat2rotm(x(:,end)'), zeros(3,1), eye(3)*.3);
% view(3); axis equal; axis tight; axis vis3d;  
% h=[];
% for t=1:nbData
% 	if mod(t,10)~=0
% 		delete(h);
% 	end
% 	[U,~,V] = svd(w(1,t) .* R1 + w(2,t) .* R2);
% 	R = U * V';
% 	h = plot3Dframe(R, zeros(3,1));
% 	drawnow;
% end

%print('-dpng','graphs/demo_Riemannian_S3_interp01.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, mu)
	x = QuatMatrix(mu) * expfct(u);
end

function u = logmap(x, mu)
	if norm(mu-[1;0;0;0])<1e-6
		Q = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
	else
		Q = QuatMatrix(mu);
	end
	u = logfct(Q'*x);
end

function Exp = expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2+u(3,:).^2);
	Exp = real([cos(normv) ; u(1,:).*sin(normv)./normv ; u(2,:).*sin(normv)./normv ; u(3,:).*sin(normv)./normv]);
	Exp(:,normv < 1e-16) = repmat([1;0;0;0],1,sum(normv < 1e-16));
end

function Log = logfct(x)
% 	scale = acos(x(3,:)) ./ sqrt(1-x(3,:).^2);
	scale = acoslog(x(1,:)) ./ sqrt(1-x(1,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(2,:).*scale; x(3,:).*scale; x(4,:).*scale];
end

function Q = QuatMatrix(q)
	Q = [q(1) -q(2) -q(3) -q(4);
	     q(2)  q(1) -q(4)  q(3);
	     q(3)  q(4)  q(1) -q(2);
	     q(4) -q(3)  q(2)  q(1)];
end					 

% Arcosine redefinition to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
function acosx = acoslog(x)
	for n=1:size(x,2)
		% sometimes abs(x) is not exactly 1.0
		if(x(n)>=1.0)
			x(n) = 1.0;
		end
		if(x(n)<=-1.0)
			x(n) = -1.0;
		end
		if(x(n)>=-1.0 && x(n)<0)
			acosx(n) = acos(x(n))-pi;
		else
			acosx(n) = acos(x(n));
		end
	end
end

function Ac = transp(g,h)
	E = [zeros(1,3); eye(3)];
	vm = QuatMatrix(g) * [0; logmap(h,g)];
	mn = norm(vm);
	if mn < 1e-10
		disp('Angle of rotation too small (<1e-10)');
		Ac = eye(3);
		return;
	end
	uv = vm / mn;
	Rpar = eye(4) - sin(mn)*(g*uv') - (1-cos(mn))*(uv*uv');	
	Ac = E' * QuatMatrix(h)' * Rpar * QuatMatrix(g) * E; %Transportation operator from g to h 
end
