function T = outprod(U,varargin)
%OUTPROD Outer vector/matrix/tensor product.
%   
%   T = OUTPROD(U1,U2) returns the outer product of the dense tensors U1
%   and U2. If U1 has size I_1x...xI_M and U2 has size J_1x...xJ_M, the
%   tensor T has size I_1x...xI_MxJ_1x...xJ_M. If U1, resp. U2, is a row or
%   column vector, the singleton dimension is discarded.
%
%   T = OUTPROD(U1,U2,...) and OUTPROD({U1,U2,...}) return the outer
%   product of the dense tensors U1 o U2 o ..., where o denotes the outer
%   product.

%   Authors: Otto Debals (Otto.Debals@esat.kuleuven.be)
%            Nico Vervliet (Nico.Vervliet@esat.kuleuven.be)
%            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
%   Version History:
%   - 2016/01/11   OD      Initial version

% Concatenate inputs
if ~iscell(U), U = {U}; end
if nargin>1
    if any(cellfun(@iscell,varargin))
        varargin = cat(2,varargin{:});
    end
    U = [U varargin{:}];
end

if numel(U)==1,
    % Return vector/matrix/tensor
    T = U{1};
else
    T = bsxfun(@times,U{1}(:),U{2}(:).');
    for i = 3:numel(U)
        T = bsxfun(@times,T(:),U{i}(:).');
    end
    % Reshape matrix
    total_size = cell2mat(cellfun(@(x) vecsize(x),U,'UniformOutput',false));
    T = reshape(T,total_size);
end
end

function s = vecsize(x)
% Return size(x) if x is a matrix/tensor, and return numel(x) if x is a vector.
if isvector(x), s = numel(x);
else s = size(x);
end
end