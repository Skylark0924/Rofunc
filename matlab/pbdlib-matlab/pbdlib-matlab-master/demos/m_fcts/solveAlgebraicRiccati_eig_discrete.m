function X = solveAlgebraicRiccati_eig_discrete(A, G, Q)
% Solves the algebraic Riccati equation (ARE) for discrete systems of the form 
% X=A'XA-(A'XB)(R+B'XB)^{-1}(B'XA)+Q with eigendecomposition, where X is symmetric.
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
% Copyright (c) 2016 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon (http://calinon.ch/)
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

n = size(A,1);

%Symplectic matrix (see https://en.wikipedia.org/wiki/Algebraic_Riccati_equation)
%Z = [A+B*(R\B')/A'*Q, -B*(R\B')/A'; -A'\Q, A'^-1]; 
Z = [A+G/A'*Q, -G/A'; -A'\Q, inv(A')]; 


%Since Z is symplectic, if it does not have any eigenvalues on the unit circle, 
%then exactly half of its eigenvalues are inside the unit circle. 
[V,D] = eig(Z); 
U = [];
for j=1:2*n
	if norm(D(j,j)) < 1 %inside unit circle
		U = [U V(:,j)];
    end
end

X = real(U(n+1:end,:) / U(1:n,:));
