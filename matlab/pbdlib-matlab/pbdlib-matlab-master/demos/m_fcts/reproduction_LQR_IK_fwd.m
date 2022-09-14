function r = reproduction_LQR_IK_fwd(DataIn, model, r, x0, rFactor, q0, robot)
%LQR in joint space and task space with task-parameterized models
%Danilo Bruno, Sylvain Calinon, 2015

nbData = size(DataIn,2);
nbVarQ = size(q0,1);
nbVarX = size(x0,1);

%% IK with LQR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = q0;
J = zeros(nbVarX*2, nbVarQ*2, nbData);
JxDot = zeros(nbVarX,nbVarQ,nbData);
A = kron([0 1; 0 0], eye(nbVarQ));
B = kron([0; 1], eye(nbVarQ));
Q = zeros(nbVarX*2,nbVarX*2,nbData);

R = eye(nbVarQ) * rFactor;

S = zeros(nbVarQ*2,nbVarX*2,nbData);
T = zeros(nbVarQ*2,nbVarQ*2,nbData);
d = zeros(nbVarQ*2,nbData);

Lx(:,:,1) = R\B' * S(:,:,1);
Lq(:,:,1) = R\B' * T(:,:,1);
ff(:,1) = R\B' * d(:,1);
KK = B*(R\B');

Jtmp = robot.jacob0(q);    
Jx(:,:,1) = Jtmp(1:2,:);

dq = zeros(nbVarQ,1);
dx = zeros(nbVarX,1);

%Update velocity and position 
Htmp = robot.fkine(q); %Forward kinematics (for each link)        
x = Htmp.t(1:2,end); %Saving Cartesian position (x,y)
%Log data
r.Data(:,1) = [DataIn(:,1); x];
r.q(:,1) = q; 
	
for t=2:nbData
	%Robot control
	Jtmp = robot.jacob0(q);    
	Jx(:,:,t) = Jtmp(1:2,:);
	JxDot(:,:,t) = (Jx(1:nbVarX,1:nbVarQ,t) - Jx(1:nbVarX,1:nbVarQ,t-1))/model.dt;
	J(:,:,t) = [Jx(:,:,t) zeros(nbVarX,nbVarQ) ; JxDot(:,:,t) Jx(:,:,t)];
	Q(1:nbVarX,1:nbVarX,t) = inv(r.currSigma(1:nbVarX,1:nbVarX,t)); %eye(nbVar)*1E0;
	%Computation of the terms in u(t) = -L(t)X(t) - M(t)q(t) + N(t)
	S(:,:,t) = S(:,:,t-1) - model.dt*(-J(:,:,t)'*Q(:,:,t) + (S(:,:,t-1)*J(:,:,t) + T(:,:,t-1))*KK*S(:,:,t-1) - A'*S(:,:,t-1));
	T(:,:,t) = T(:,:,t-1) - model.dt*(-(S(:,:,t-1)*J(:,:,t) + T(:,:,t-1))*A + (S(:,:,t-1)*J(:,:,t)+T(:,:,t-1))*KK*T(:,:,t-1) - A'*T(:,:,t-1));
	d(:,t) = d(:,t-1) - model.dt*((S(:,:,t-1)*J(:,:,t)+T(:,:,t-1))*KK*d(:,t-1) - S(:,:,t-1)*[r.currTar(nbVarX+1:2*nbVarX,t); zeros(nbVarX,1)] - A'*d(:,t-1) );
	Lx(:,:,t) = R\B' * S(:,:,t);
	Lq(:,:,t) = R\B' * T(:,:,t);
	ff(:,t) = R\B' * d(:,t);
  
	%ddq =  -Lx(:,:,t) * [x-r.currTar(1:2,t); dx-r.currTar(3:4,t)] - Lq(:,:,t) * [q;dq] + ff(:,t);
	ddq =  -Lx(:,:,t) * [x-r.currTar(1:2,t); dx] - Lq(:,:,t) * [q;dq] + ff(:,t);
	%Update velocity and position
	dq = dq + ddq * model.dt;
	%dx = J(1:nbVarX,1:nbVarQ,t) * dq;
	q = q + dq * model.dt;
	%x = x + dx * model.dt;
	Htmp = robot.fkine(q); %Forward kinematics (for each link)        
	x_new = Htmp.t(1:2,end); %Saving Cartesian position (x,y)
	dx = (x_new-x)/model.dt;
	x = x_new;

	%Log data
	r.Data(:,t) = [DataIn(:,t); x];
	r.q(:,t) = q; 
end