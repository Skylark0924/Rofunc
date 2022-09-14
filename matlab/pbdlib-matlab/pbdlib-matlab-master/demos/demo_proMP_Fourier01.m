function demo_proMP_Fourier01
% ProMP with Fourier basis functions (1D example).
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
% 	author="Calinon, S.",
% 	title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
% 	booktitle="Mixture Models and Applications",
% 	publisher="Springer",
% 	editor="Bouguila, N. and Fan, W.", 
% 	year="2019",
% 	pages="39--57",
% 	doi="10.1007/978-3-030-23876-6_3"
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 200; %Number of datapoints in a trajectory
nbSamples = 4; %Number of demonstrations


%% Generate periodic data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbSamples
	t = linspace(n/3, 4*pi+n/3, nbData);
	x(:,n) = (1+n.*1E-1) .* cos(t) + (.4+n.*2E-1) * cos(t*2+pi/3) + randn(1,nbData) .* 1E-4;
end


%% ProMP with Fourier basis functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute basis functions Psi and activation weights w
k = -5:5;
nbFct = length(k);
t = linspace(0,1,nbData);
Psi = exp(t' * k * 2 * pi * 1i); % / nbData;
% w = (Psi' * Psi + eye(nbStates).*1E-18) \ Psi' * x; 
w = pinv(Psi) * x;

%Distribution in parameter space
Mu_R = mean(abs(w), 2); %Magnitude average
Mu_theta = mean_angle(angle(w), 2); %Phase average
Mu_w = Mu_R .* exp(1i * Mu_theta); %Reconstruction

Sigma_R = cov(abs(w')); %Magnitude spread
Sigma_theta = cov_angle(angle(w')); %Phase spread
Sigma_w = Sigma_R .* exp(1i * Sigma_theta)  + eye(size(Sigma_R)) * 1E-2; %Reconstruction

%Trajectory distribution
Mu = Psi * Mu_w; 
Sigma = Psi * Sigma_w * Psi'; 


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(nbFct);
figure('position',[10 10 800 800]); 
%Plot signal
subplot(3,1,1); hold on; %axis off;
for n=1:nbSamples 
	plot(x(:,n), '-','lineWidth',5,'color',[.7 .7 .7]);
end
% std = real(diag(Sigma)'.^.5) .* 3;
std = real(diag(sqrtm(Sigma))') .* 30;
% std = diag(sqrtm(real(Sigma))') .* 30;
msh = [[1:nbData, nbData:-1:1]; [real(Mu')+std, fliplr(real(Mu')-std)]];
patch(msh(1,:), msh(2,:), [.8 0 0],'edgecolor','none','facealpha',.3); 
h(1) = plot(real(Mu), '-','lineWidth',3,'color',[.8 0 0]);
h(2) = plot(imag(Mu), ':','lineWidth',3,'color',[.8 0 0]);
axis([1, nbData, min(msh(2,:))-.5, max(msh(2,:))+.5]); %axis tight;
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',25); ylabel('$x$','interpreter','latex','fontsize',25);
legend(h,{'Re($\mu^x$)','Im($\mu^x$)'},'interpreter','latex','location','southeast','fontsize',22);
%Plot basis functions
subplot(3,1,2); hold on; %axis off; %real part
for i=1:nbFct
	plot(1:nbData, real(Psi(:,i)),'-','linewidth',3,'color',clrmap(i,:)); %axis tight; 
end
axis([1, nbData, min(real(Psi(:))), max(real(Psi(:)))]);
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',25); ylabel('Re($\phi_k$)','interpreter','latex','fontsize',25);

subplot(3,1,3); hold on; %axis off; %imaginary part
for i=1:nbFct
	plot(1:nbData, imag(Psi(:,i)),':','linewidth',3,'color',clrmap(i,:));
end
axis([1, nbData, min(imag(Psi(:)))-1E-4, max(imag(Psi(:)))+1E-4]); %axis tight; 
set(gca,'xtick',[],'ytick',[]);
xlabel('$t$','interpreter','latex','fontsize',25); ylabel('Im($\phi_k$)','interpreter','latex','fontsize',25);

%Plot magnitude and phase 
figure('position',[820 10 800 800]); hold on; axis off;
plot2DArrow([0;0], [.7;0].*nbData, [0,0,0], 1, .03.*nbData);
plot2DArrow([0;0], [0;.7].*nbData, [0,0,0], 1, .03.*nbData);
text((.7-.15).*nbData, .05.*nbData, 'Re($w$)','interpreter','latex','fontsize',20); text(.02.*nbData, (.7-.1).*nbData, 'Im($w$)','interpreter','latex','fontsize',20);
nbDrawingSeg = 50;
t = linspace(-pi, pi, nbDrawingSeg);
xc = [cos(t); sin(t)];
for i=1:nbFct
% 	msh = [];
% 	for j=linspace(0,2,nbDrawingSeg)
% 		msh = [msh, (Mu_R(i)-Sigma_R(i,i)^.5) .* exp(1i * (Mu_theta(i) - Sigma_theta(i,i)^.5 + j * Sigma_theta(i,i)^.5))]; 
% 	end
% 	for j=linspace(2,0,nbDrawingSeg)
% 		msh = [msh, (Mu_R(i)+Sigma_R(i,i)^.5) .* exp(1i * (Mu_theta(i) - Sigma_theta(i,i)^.5 + j * Sigma_theta(i,i)^.5))]; 
% 	end
% 	patch(real(msh), imag(msh), clrmap(i,:),'edgecolor','none','facealpha',.1);
	R = diag([Sigma_theta(i,i).^.5, Sigma_R(i,i).^.5]) + eye(2).*1E-6;
	x = R * xc + repmat([Mu_theta(i); Mu_R(i)], 1, nbDrawingSeg);
	[x(1,:), x(2,:)] = pol2cart(x(1,:), x(2,:));
	patch(x(1,:), x(2,:), clrmap(i,:),'edgecolor','none','facealpha',.3);
end
for i=1:nbFct
	plot(real(w(i,:)), imag(w(i,:)), '.','markersize',25,'color',clrmap(i,:));
	for j=1:size(w,2)
		patch([0 real(w(i,j)) 0], [0 imag(w(i,j)) 0], clrmap(i,:), 'edgecolor',clrmap(i,:),'facealpha',.3,'edgealpha',.3);
	end
	plot(real(Mu_w(i)), imag(Mu_w(i)), '.','markersize',25,'color',clrmap(i,:));
	plot(real(Mu_w(i)), imag(Mu_w(i)), 'o','markersize',6,'linewidth',2,'color',clrmap(i,:));
	patch([0 real(Mu_w(i)) 0], [0 imag(Mu_w(i)) 0], clrmap(i,:), 'linewidth',4,'edgecolor',clrmap(i,:),'facealpha',.3,'edgealpha',.3);
end
axis equal;

% print('-dpng','graphs/proMP_Fourier01.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%
function Mu = mean_angle(phi, dim)
	if nargin<2
		dim = 1;
	end
	Mu = angle(mean(exp(1i*phi), dim));
end

%%%%%%%%%%%%%%%%%%
function Sigma = cov_angle(phi)
	Mu = mean_angle(phi);
	e = phi - repmat(Mu, size(phi,1), 1);
	Sigma = cov(angle(exp(1i*e)));
end
