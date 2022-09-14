function demo_GMR03
% Chain rule with Gaussian conditioning
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
model.nbVar = 3; %Number of variables [t,x1,x2]
nbData = 20; %Length of each trajectory


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = rand(model.nbVar, nbData);
model.Mu = mean(Data,2);
model.Sigma = cov(Data');


%% Chain rule
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=1;
gaussPDF(Data(:,t), model.Mu, model.Sigma)

p=1;
for i=1:model.nbVar
	in = 1:i-1;
	out = i;
	%Gaussian conditioning
	MuOut = model.Mu(out) + model.Sigma(out,in)/model.Sigma(in,in) * (Data(in,t) - model.Mu(in));
	SigmaOut = model.Sigma(out,out) - model.Sigma(out,in)/model.Sigma(in,in) * model.Sigma(in,out);
	p = p * gaussPDF(Data(out,t), MuOut, SigmaOut);
end
p