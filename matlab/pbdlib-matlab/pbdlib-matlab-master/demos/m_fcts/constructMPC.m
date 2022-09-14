function [Phi,F] = constructMPC(Ap,Bp,Cp,Nc,Np)
%Construct Phi matrix and F vector for model predictive control
%Martijn Zeestraten, 2015

% %Determine size of system variables:
% [m1,~]	= size(Cp);
% [n1,n_in]	= size(Bp);
%
% % Construct augmented system:
% A_e				= eye(n1+m1,n1+m1);
% A_e(1:n1,1:n1)	= Ap;
% A_e(n1+1:n1+m1,1:n1) = Cp*Ap;
%
% B_e			= zeros(n1+m1,n_in);
% B_e(1:n1,:) = Bp;
% B_e(n1+1:n1+m1,:) = Cp*Bp;
%
% C_e	= zeros(m1,n1+m1);
% C_e(:,n1+1:n1+m1) = eye(m1,m1);

nC = size(Cp,1);
mA = size(Ap,2);
mB = size(Bp,2);

%Construct F vector
F  = zeros(nC*Np,mA);
c1 = zeros(nC*Np,mB);
F(1:nC,:) = Cp*Ap;
%F(1:nC,:) = Cp*eye(mA); %Syl

c1(1:nC,:) = Cp*Bp;
for kk=2:Np
	ind1 = ((kk-2)*nC+1):(kk-1)*nC;
	ind2 = ((kk-1)*nC+1):kk*nC;
	F(ind2,:) = F(ind1,:)*Ap;
	c1(ind2,:) = F(ind1,:)*Bp; 
	%c1(ind2,:) = F(ind2,:)*Bp; %Syl
end

%Construct Phi matrix
Phi = zeros(Np*nC,mB*Nc);
Phi(:,1:mB) = c1;
for kk=2:Nc
	rInd1 = (kk-1)*nC+1;
	rInd2 = (Np-kk+1)*nC;
	cInd = ((kk-1)*mB+1):kk*mB;
	Phi(rInd1:end,cInd) = c1(1:rInd2,:);
end

% Phi
% pcolor(Phi)
% pause
% close all;
