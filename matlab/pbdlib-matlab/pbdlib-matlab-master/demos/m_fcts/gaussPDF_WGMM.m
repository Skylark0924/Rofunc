function prob = gaussPDF_WGMM(Data, Mu, Sigma, v)
% Likelihood of a wrapped Gaussian distribution encoding partially periodic signals, with periodic dimensions defined in 'v'
% Sylvain Calinon, 2015

[nbVar,nbData] = size(Data);
for j=1:nbVar
	nr(j) = length(v(j).rg);
end

%Fast version (see readable but slow 2D step-by-step version below)
Xtmp = computeHypergrid_WGMM(v, nbData);
DataTmp = (repmat(Data,1,prod(nr)) - Xtmp - repmat(Mu,1,nbData*prod(nr)))';
probTmp = sum((DataTmp/Sigma).*DataTmp, 2);
prob = sum(reshape(exp(-0.5*probTmp),nbData,prod(nr)),2) / sqrt((2*pi)^nbVar * (abs(det(Sigma))+realmin));

% %Slow 2D step-by-step version (easier to read)
% prob = zeros(nbData,1);
% for j=1:length(v(1).rg)
%   for k=1:length(v(2).rg)
%     DataTmp = Data' - repmat(Mu'+[v(1).rg(j),v(2).rg(k)],nbData,1);
%     probTmp = sum((DataTmp/Sigma).*DataTmp, 2);
%     prob = prob + exp(-0.5*probTmp) / sqrt((2*pi)^nbVar * (abs(det(Sigma))+realmin));
%   end
% end