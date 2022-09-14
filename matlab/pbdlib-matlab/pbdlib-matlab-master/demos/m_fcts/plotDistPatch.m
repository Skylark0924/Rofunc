function h=plotDistPatch(X,Mu,Var,color,varargin)
% Martijn Zeestraten, 2015
% Function plots a 1D patch of a state 
% Input: 
% X    :   [1x nbData] Vector of X values
% Mu   :   [1x nbData] Mean value of patch
% Sigma:   [1x nbData] Variance of Patch 
% Color:   [1x 3     ] Color vector
% nbStd:   (optional)  Number of standard deviations for the
%                      patch
% varargin:            Optimal arguments to modify the appearance of the
%                      patch/line

% Default Settings:
LW        = 2;
FaceAlpha = 0.4;
nbStd     = 2;


%% Handle Additional arguments:
if nargin>4 
	% Assign additional arguments to patch and/or line if these apply
	
	% Create test objects
	tpl =plot(1,1);
	tpa =patch(1,1,1);
	
	% Pre-allocate space to speed up process	
	LineArgin  =cell(size(varargin));
	PatchArgin =cell(size(varargin));
	
	% Counters to keep track on the amount of properties
	lineCount = 0;	patchCount =0;
	
	% For half the number of extra argument (each argument should come in
	% two, name and value)
	for i = 1:((nargin-3)/2)
		
		% Check if it belongs to plot properties
		if isprop(tpl,varargin{(i-1)*2+1})==1
			LineArgin{lineCount+1} = varargin{(i-1)*2+1};
			LineArgin{lineCount+2} = varargin{(i-1)*2+2};
			lineCount = lineCount+2;
		end
				
		% Check if it belongs to patch properties
		if isprop(tpa,varargin{(i-1)*2+1}) ==1
			PatchArgin{patchCount+1} = varargin{(i-1)*2+1};
			PatchArgin{patchCount+2} = varargin{(i-1)*2+2};
			patchCount = patchCount+2;
        end
        
        if strcmp('nbStd',varargin{(i-1)*2+1})==1
           % Number of standard deviations
           nbStd =  varargin{(i-1)*2+2};
        end
		
	end
end

%% Create Patch Entries
vTmp = sqrt(Var)*nbStd; 
msh  = [X,fliplr(X);
	    Mu+vTmp,fliplr(Mu-vTmp)];
  
%% Plot Patch:   
if nargin ==4
	% Patch:
	patch(msh(1,:),msh(2,:), color, 'EdgeColor', 'none',...
					'FaceAlpha', FaceAlpha, 'LineWidth', LW);
	% Center line:
	h=plot(X,Mu,'color', color,'LineWidth', LW/2);
		
else
	% Patch:
	patch(msh(1,:),msh(2,:),color,'EdgeColor', 'none', ...
					'FaceAlpha', FaceAlpha, 'LineWidth', LW,LineArgin{1:lineCount});
	% Center line:
	h=plot(X,Mu,'color', color,'LineWidth', LW/2,PatchArgin{1:patchCount});	
end
			
end