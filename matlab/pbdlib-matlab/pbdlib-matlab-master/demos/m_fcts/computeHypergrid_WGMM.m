function X = computeHypergrid_WGMM(v, nbData)
%This function creates an N-dimension hypergrid from the range parameters
%from 'v', with a repetition pattern defined by 'nbData'
%It is an extension of the ndgrid Matlab function.
%Author: Sylvain Calinon, 2013

nbVar = length(v);
for j=1:nbVar
	nr(j) = length(v(j).rg);
end

if nbVar>1 %Requires hypergrid
	for j=1:nbVar
		s=nr; s(j)=[]; %Remove i-th dimension (see ndgrid function)
		x=v(j).rg(:); x=reshape(x(:,ones(1,prod(s))),[length(x) s]); %Expand x (see ndgrid function)
		X(j,:) = reshape(repmat(reshape(permute(x,[2:j 1 j+1:nbVar]),1,prod(nr)),nbData,1),1,nbData*prod(nr)); %See ndgrid function
	end
else %1D case
	X(1,:) = repmat(v(1).rg,1,nbData);
end
