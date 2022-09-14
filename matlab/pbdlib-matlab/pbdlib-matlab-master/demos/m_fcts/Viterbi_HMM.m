function q = Viterbi_HMM(Data, model)
% Viterbi path decoding (MAP estimate of best path) in HMM.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Rozo16Frontiers,
%   author="Rozo, L. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
%   title="Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration",
%   journal="Frontiers in Robotics and {AI}",
%   year="2016",
%   month="June",
%   volume="3",
%   number="30",
%   pages="1--11",
%   doi="10.3389/frobt.2016.00030"
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


nbData = size(Data,2);


% %MPE estimate of best path (for comparison)
% H = computeGammaHMM(s(1), model);
% [~,q] = max(H);


% %Viterbi with scaling factor to avoid numerical underflow
% for i=1:model.nbStates
% 	B(i,:) = gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i)); %Emission probability
% end
% %Viterbi forward pass
% DELTA(:,1) = model.StatesPriors .* B(:,1);
% DELTA(:,1) = DELTA(:,1) / norm(DELTA(:,1)); %Scaled version of Delta to avoid numerical underflow 
% PSI(1:model.nbStates,1) = 0;
% for t=2:nbData
% 	for i=1:model.nbStates
% 		[maxTmp, PSI(i,t)] = max(DELTA(:,t-1) .* model.Trans(:,i));
% 		DELTA(i,t) = maxTmp * B(i,t); 
% 	end
% 	DELTA(:,t) = DELTA(:,t) / norm(DELTA(:,t)); %Scaled version of Delta to avoid numerical underflow 
% end
% %Backtracking
% q = [];
% [~,q(nbData)] = max(DELTA(:,nbData));
% for t=nbData-1:-1:1
% 	q(t) = PSI(q(t+1),t+1);
% end


%Viterbi with log computation
for i=1:model.nbStates
	logB(i,:) = log(gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i))); %Emission probability
end
%Viterbi forward pass
logDELTA(:,1) = log(model.StatesPriors) + logB(:,1);
PSI(1:model.nbStates,1) = 0;
for t=2:nbData
	for i=1:model.nbStates
		[maxTmp, PSI(i,t)] = max(logDELTA(:,t-1) + log(model.Trans(:,i)));
		logDELTA(i,t) = maxTmp + logB(i,t); 
	end
end
%Backtracking
q = [];
[~,q(nbData)] = max(logDELTA(:,nbData));
for t=nbData-1:-1:1
	q(t) = PSI(q(t+1),t+1);
end
