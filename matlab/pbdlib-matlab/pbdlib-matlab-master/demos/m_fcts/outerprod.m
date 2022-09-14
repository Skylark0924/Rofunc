function T = outerprod(U,V,varargin)
% Computation of the outer product between two tensors.

T = bsxfun(@times,U(:),V(:).');
totalSize = [size(U) size(V)];

for i = 1:numel(varargin)
	T = bsxfun(@times,T(:),varargin{i}(:).');
	totalSize = [totalSize, size(varargin{i})];
end

totalSize(totalSize==1) = [];

T = reshape(T, totalSize);

end