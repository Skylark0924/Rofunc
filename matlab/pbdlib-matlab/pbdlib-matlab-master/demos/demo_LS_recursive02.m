function demo_LS_recursive02
% Recursive computation of least squares estimate (with one datapoint at a time)
% (Implementation based on Hager 1989 "Updating the Inverse of a Matrix", Y=XA -> Y'=A'X') 
% 
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarIn = 3; %Dimension of input vector
nbVarOut = 1; %Dimension of output vector
nbData = 20; %Number of observations


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A0 = rand(nbVarIn,nbVarOut); %Linear relation between input and output (to be estimated)
X = rand(nbData,nbVarIn); %Input data
Y = X * A0 + randn(nbData,nbVarOut)*1E-2; %Output data (with noise)
%X = repmat(linspace(0,1,nbData)',1,nbVarIn); %Input data
%Y = sin(X*pi*2) + X*5 + randn(nbData,nbVarOut)*1E-10; %Output data (with noise)


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Batch estimate (with pseudoinverse)
tic
A = (X'*X \ X') * Y
% A = X \ Y
t=toc; disp(['Standard computation:       ' num2str(t) ' sec.']);
%Compute residuals of LS
r0 = 0;
for t=1:nbData
	r0 = r0 + norm(Y(t,:)-X(t,:)*A, 2);
end
disp(['Error with ordinary least squares               : ' num2str(r0)]);

%Incremental estimate (does not require matrix inversion)
tic
A = zeros(nbVarIn,nbVarOut); %Initial estimate of A
iB = eye(nbVarIn) * 1E10; %Initial estimate of iB
for t=1:nbData
	V = X(t,:); %New input data
	C = Y(t,:); %New output data 
	K =  iB*V' ./ (1 + V*iB*V'); %Kalman gain
	A = A + K * (C-V*A); %Update A
	iB = iB - iB*V' ./ (1+V*iB*V')*V*iB; %Update iB
	%disp(['Error on incremental estimate: ' num2str(norm(A-A0))]);
	s(t).A = A; %Log estimate
end
t=toc; disp(['Recursive computation: ' num2str(t) ' sec.']);
%Compute residuals of LS
r = 0;
for t=1:nbData
	r = r + norm(Y(t,:)-X(t,:)*A, 2);
end
disp(['Error with recursive least squares               : ' num2str(r)]);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 4],'position',[10,10,1200,500]);
i=1; j=1;
%LS
subplot(1,2,1); hold on; title(['Ordinary least squares (e=' num2str(r0,'%.1f') ')']);
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',[0 0 0]);
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot([0 1], [0 A(i,j)], 'r-','linewidth',2);
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -1 9]);
%RLS
subplot(1,2,2); hold on; title(['Recursive least squares (e=' num2str(r,'%.1f') ')']);
for t=4:2:nbData-4
	plot([0 1], [0 s(t).A(i,j)], '-','linewidth',2,'color',[1 1-(t/nbData)^2 1-(t/nbData)^2]);
end
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',[0 0 0]);
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot([0 1], [0 s(end).A(i,j)], '-','linewidth',2,'color',[1 0 0]);
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -1 9]);

%print('-dpng','-r600','graphs/demo_LS_recursive02.png');
pause;
close all;