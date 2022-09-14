function demo_LS_IRLS01
% Iteratively reweighted least squares (IRLS)
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
nbD = 20; %Number of datapoints in each dimension
nbData = nbD^nbVarIn; %Number of datapoints
p = 1; %Lp norm
nbIter = 20; %Number of iterations for IRLS
regTerm = 0E-1; %Regularization term for the reweighting


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = rand(nbData,nbVarIn); %Input data
% X = repmat(linspace(0,1,nbData)', 1, nbVarIn); %Input data
X = ndarr(linspace(0,1,nbD), nbVarIn);

%Y = sin(X*pi*2) + X*5 + randn(nbData,nbVarOut)*1E-10; %Output data (with noise)
Ad = rand(nbVarIn,nbVarOut) + 4;
Y = X * Ad + randn(nbData, nbVarOut) * 1E-2; %Output data (with noise) %rand(nbVarIn,nbVarOut)

Y(4,:) = Y(4,:) - 3; %Simulation of outlier
Y(17,:) = Y(17,:) - 5; %Simulation of outlier


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization (with each datapoint having the same weight)
W = eye(nbData);
A0 = X'*W*X \ X'*W * Y;

%Compute initial residuals of LS with Lp-norm error
r0 = 0;
for t=1:nbData
	r0 = r0 + norm(Y(t,:)-X(t,:)*A0, p);
end
	
%Iteratively reweighted least squares
for n=1:nbIter
	A = X'*W*X \ X'*W * Y; %Weighted least squares estimate
	
% 	for t=1:nbData
% 		W(t,t) = sum(abs(Y(t,:)-X(t,:)*A) + regTerm).^(p-2); %Reweighting of the datapoints (sum() is used here if nbVarOut>1)
% 	end
	w = abs(Y-X*A + regTerm).^(p-2);
	W = diag(w);	
	
	rn(n).W = W; %Log data
	
% 	%Compute residuals of IRLS
% 	rn(n) = 0;
% 	for t=1:nbData
% 		rn(n) = rn(n) + norm(Y(t,:)-X(t,:)*A, p);
% 	end
end

%Compute residuals of IRLS with Lp-norm error
r = 0;
for t=1:nbData
	r = r + norm(Y(t,:)-X(t,:)*A, p);
end

disp(['Lp-norm error with ordinary least squares               : ' num2str(r0)]);
disp(['Lp-norm error with iteratively reweighted least squares : ' num2str(r)]);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 4],'position',[10,10,1200,500]);
%Normalize W for display purpose
W = W - min(diag(W));
W = W / max(diag(W));
i=1; j=1; %Dimensions used for plots
%LS
subplot(1,2,1); hold on; title(['Ordinary least squares (e=' num2str(r0,'%.1f') ')']);
plot([0 1], [0 A0(i,j)], 'r-','linewidth',2);
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A0(i,j)], '-','linewidth',2,'color',[0 0 0]);
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -3 6]);
%IRLS
subplot(1,2,2); hold on; title(['Iteratively reweighted least squares (e=' num2str(r,'%.1f') ')']);
plot([0 1], [0 A(i,j)], 'r-','linewidth',2);
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',ones(1,3)*(0.9-W(t,t)*0.9));
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',ones(1,3)*(0.9-W(t,t)*0.9));
end
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -3 6]);

% %Plot the importance of each datapoint
% figure; 
% subplot(2,1,1); hold on;
% for n=1:nbIter
% 	plot(diag(rn(n).W),'.','markersize',10,'color',[1,1,1]-n/nbIter);
% end
% xlabel('n'); ylabel('w');
% subplot(2,1,2); hold on;
% plot(diag(W),'r.','markersize',15);
% xlabel('n'); ylabel('w');

% %Plot the error
% figure; hold on;
% plot(rn,'k-');
% plot([1,nbIter], [r0,r0], 'r-');
% xlabel('n'); ylabel('e');

%print('-dpng','-r600','graphs/demo_LS_IRLS01.png');
pause;
close all;
end


function x = ndarr(lst, nbVar)
	x = [];
	for n=1:nbVar
		s = ones(1,nbVar); 
		s(n) = numel(lst);
		if nbVar>1
			lst = reshape(lst,s);
		end
		s = repmat(numel(lst),1,nbVar); 
		s(n) = 1;
		xtmp = repmat(lst,s);
		x = [x, xtmp(:)];
% 		x = cat(n+1,x,repmat(lst,s));
	end
end