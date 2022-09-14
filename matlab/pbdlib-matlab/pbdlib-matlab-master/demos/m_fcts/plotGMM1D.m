function [h,Xout] = plotGMM1D(model, boundbox, color, valAlpha, nbData, orient_option)
% Plot GMM profile horizontally or vertically
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

if nargin<2
	boundbox = [-1,1,0,1];
end
if nargin<3
	color = [0,0,0];
end
if nargin<4
	valAlpha = 0.6;
end
if nargin<5
	nbData = 200;
end
if nargin<6
	orient_option = 'h';
end

if ~isfield(model,'nbStates')
	model.nbStates = size(model.Mu,2);
end
if ~isfield(model,'Priors')
% 	model.Priors = ones(model.nbStates,1);
	model.Priors = ones(model.nbStates,1) / model.nbStates;
end

lightcolor = min(color+0,1);

X=[];
for i=1:model.nbStates
	Xin(1,:) = linspace(boundbox(1), boundbox(2), nbData);
	X = [X [Xin; model.Priors(i)*gaussPDF(Xin, model.Mu(:,i), model.Sigma(:,:,i))]];
end

Xout = sum(reshape(X(2,:),nbData,model.nbStates),2)';

%Fit the data to the bounding box (optional)
X(2,:) = X(2,:) / max(Xout);
Xout = Xout / max(Xout);

if orient_option=='h'
	P = [X(1,:); boundbox(3)+(boundbox(4)-boundbox(3))*X(2,:)]; 
else
	P = [boundbox(3)+(boundbox(4)-boundbox(3))*X(2,:); X(1,:)]; 
end

%h = patch([P(1,1) P(1,:) P(1,end)], [boundbox(2) P(2,:) boundbox(2)], ...
%		[.7 .7 .7], 'lineWidth', 1, 'EdgeColor', [.2 .2 .2], 'facealpha', valAlpha,'edgealpha', valAlpha);
h = [];
% h = [h plot(X(1,:), X(2,:), '-','linewidth',4,'color',[1 .7 .7])];
for i=1:model.nbStates
% 	h = [h patch([X(1,(i-1)*nbData+1) X(1,(i-1)*nbData+1:i*nbData) X(1,i*nbData)], ...
% 		[boundbox(3) X(2,(i-1)*nbData+1:i*nbData) boundbox(3)], ...
% 		lightcolor, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
	if orient_option=='h'
		h = [h patch([P(1,(i-1)*nbData+1) P(1,(i-1)*nbData+1:i*nbData) P(1,i*nbData)], ...
			[boundbox(3) P(2,(i-1)*nbData+1:i*nbData) boundbox(3)], ...
			lightcolor, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
	else
		h = [h patch([boundbox(3) P(1,(i-1)*nbData+1:i*nbData) boundbox(3)], ...
			[P(2,(i-1)*nbData+1) P(2,(i-1)*nbData+1:i*nbData) P(2,i*nbData)], ...
			lightcolor, 'lineWidth', 1, 'EdgeColor', color, 'facealpha', valAlpha,'edgealpha', valAlpha)];
	end
end
% h = plot(P(1,:), P(2,:), '-','linewidth',2,'color',[.8 0 0]);
