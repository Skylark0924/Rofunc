function [Phi,F] = constructTrajMPC(Ap,Bp,Nc,Np)
% Martijn Zeestraten, May 2015
% Code based on book:
% @BOOK{Wang09,
%   title = {Model predictive control system design and implementation using MATLAB{\textregistered}},
%   publisher = {Springer Science \& Business Media},
%   year = {2009},
%   author = {Wang, Liuping},
%   file = {:/home/martijn/Dropbox/PhD/Literature/literature_repository/Books/Wang, Liuping - Model Predictive Control System Design and Implementation Using MATLAB (2010).pdf:PDF}
% }
%
% SC: Redundant with constructMPC()?


[nA,mA] = size(Ap);
[~,mB] = size(Bp);

% Construct F vector:
F  = zeros(nA*Np,mA);
c1 = zeros(nA*Np,mB);
F(1:nA,:) = Ap;
c1(1:nA,:) = Bp;
for kk = 2:Np	
	ind1 = ((kk-2)*nA+1):(kk-1)*nA;
	ind2 = ((kk-1)*nA+1):kk*nA;
	
	F(ind2,:) = F(ind1,:)*Ap;	
	c1(ind2,:) = F(ind1,:)*Bp;
end

Phi = zeros(Np*nA,mB*Nc);

Phi(:,1:mB) = c1;
for kk = 2:Nc
	rInd1 = (kk-1)*nA+1;	
	rInd2 = (Np-kk+1)*nA;	
	cInd  = ((kk-1)*mB+1):kk*mB;
	Phi(rInd1:end,cInd) = c1(1:rInd2,:);
end
