function demo_LS_weighted01
% Weighted least squares regression
% 
% If this code is useful for your research, please cite the related reference:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
% }
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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
nbVarIn = 1; %Dimension of input vector
nbVarOut = 1; %Dimension of output vector
nbData = 20; %Number of observations


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%X = rand(nbData,nbVarIn); %Input data
X = repmat(linspace(0,1,nbData)',1,nbVarIn); %Input data
Y = sin(X*pi*2) + X*5 + randn(nbData,nbVarOut)*1E-10; %Output data (with noise)
%W = eye(nbData);
W = diag(cos(linspace(0,pi/2,nbData)).^12);
%W = diag(exp(-2*X));

W = W / max(diag(W));

% W = eye(nbData)*1E-2;
% for t=1:nbData
% 	if X(t,1)<0.4
% 		W(t,t)=1;
% 	end
% end


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Least squares estimate
A = X'*X \ X' * Y;

%Weighted least squares estimate
Aw = X'*W*X \ X'*W * Y;

% %Weighted least squares estimate (computation with SVD decomposition)
% [U,S,V] = svd(W.^.5*X);
% Aw = V*pinv(S)*U' * W.^.5*Y;


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2200,900]);
i=1; j=1;
%LS
subplot(1,2,1); hold on; title('Ordinary least squares');
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',[0 0 0]);
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot([0 1], [0 A(i,j)], 'r-','linewidth',2);
%xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
xlabel('x','fontsize',16);
ylabel('y','fontsize',16);
axis([0 1 -1 9]);
%WLS
subplot(1,2,2); hold on; title('Weighted least squares');
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*Aw(i,j)], '-','linewidth',2,'color',ones(1,3)*(0.9-W(t,t)*0.9));
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',ones(1,3)*(0.9-W(t,t)*0.9));
end
plot([0 1], [0 Aw(i,j)], 'r-','linewidth',2);
%xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
xlabel('x','fontsize',16);
ylabel('y','fontsize',16);
axis([0 1 -1 9]);

%print('-dpng','graphs/demo_LS_weighted01.png');
pause;
close all;