function [PHI, PHI1, PHI0] = constructPHI(model, nbData, nbSamples)
% Construct PHI operator (big sparse matrix) used in trajectory-GMM.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
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


% op1D = zeros(model.nbDeriv);
% op1D(1,end) = 1;
% for i=2:model.nbDeriv
% 	op1D(i,:) = (op1D(i-1,:) - circshift(op1D(i-1,:),[0,-1])) / model.dt;
% end
% op = zeros(nbData*model.nbDeriv, nbData);
% op((model.nbDeriv-1)*model.nbDeriv+1:model.nbDeriv*model.nbDeriv, 1:model.nbDeriv) = op1D;
% PHI0 = zeros(nbData*model.nbDeriv, nbData);
% for t=0:nbData-model.nbDeriv
% 	PHI0 = PHI0 + circshift(op, [model.nbDeriv*t,t]);
% end
% %Handling of borders
% for i=1:model.nbDeriv-1
% 	op(model.nbDeriv*model.nbDeriv+1-i,:)=0; op(:,i)=0;
% 	PHI0 = PHI0 + circshift(op, [-i*model.nbDeriv,-i]);
% end
% %Application to multiple dimensions and multiple demonstrations
% PHI1 = kron(PHI0, eye(model.nbVarPos));
% PHI = kron(eye(nbSamples), PHI1);


%Create gradient operator template
op = sparse(nbData*model.nbDeriv, nbData);
op(1,1) = 1;
for i=2:model.nbDeriv
	op(i,:) = (op(i-1,:) - circshift(op(i-1,:),[0,-1])) / model.dt;
end
%PHI0 matrix (without border condition)
PHI0 = sparse(nbData*model.nbDeriv, nbData);
for t=0:nbData-1
	PHI0 = PHI0 + circshift(op, [model.nbDeriv*t,t]);
end
%Handling of borders
for i=1:model.nbDeriv-1
	for j=1:i
		PHI0(1+i+(j-1)*model.nbDeriv,:) = 0; %Removal of borders
	end
end
%Application to multiple dimensions
PHI1 = kron(PHI0, speye(model.nbVarPos));
PHI = kron(speye(nbSamples), PHI1);

% %Visualize PHI matrix
% figure('PaperPosition',[0 0 4 8],'position',[10 10 400 650],'name','PHI1'); 
% axes('Position',[0.01 0.01 .98 .98]); hold on; set(gca,'linewidth',2); 
% colormap(flipud(gray));
% pcolor([abs(PHI0) zeros(size(PHI0,1),1); zeros(1,size(PHI0,2)+1)]); %dummy values for correct display
% %pcolor([abs(PHI1) zeros(size(PHI1,1),1); zeros(1,size(PHI1,2)+1)]); %dummy values for correct display
% shading flat; axis ij; axis equal tight;
% set(gca,'xtick',[],'ytick',[]);



