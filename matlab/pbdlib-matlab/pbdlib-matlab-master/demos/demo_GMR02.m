function demo_GMR02
% GMR computed with precision matrices instead of covariances
% 
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 3; %Number of states in the GMM
model.nbVar = 3; %Number of variables [t,x1,x2]
nbData = 20; %Length of each trajectory


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
nbSamples = length(demos);
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data [1:nbData; s(n).Data]]; 
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);
in=1; out=2:3;
%[DataOut, SigmaOut] = GMR(model, DataIn, in, out);
%DataOut
DataIn = Data(in,:);
t=1;
for i=1:model.nbStates
	model.P(:,:,i) = inv(model.Sigma(:,:,i));
	%Regression based on covariance matrices
	MuOut = model.Mu(out,i) + model.Sigma(out,in,i)/model.Sigma(in,in,i) * (DataIn(:,t)-model.Mu(in,i))
	SigmaOut = model.Sigma(out,out,i) - model.Sigma(out,in,i)/model.Sigma(in,in,i) * model.Sigma(in,out,i)
	%Regression based on precision matrices
	MuOut = model.Mu(out,i) - model.P(out,out,i)\model.P(out,in,i) * (DataIn(:,t)-model.Mu(in,i))
	SigmaOut = inv(model.P(out,out,i))
end