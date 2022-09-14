function X = solveAlgebraicRiccati_Schur(A, G, Q)
% Solves the algebraic Riccati equation of the form A'X+XA'-XGX+Q=0, where X is symmetric.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @inproceedings{Calinon14ICRA,
%   author="Calinon, S. and Bruno, D. and Caldwell, D. G.",
%   title="A task-parameterized probabilistic model with minimal intervention control",
%   booktitle="Proc. {IEEE} Intl Conf. on Robotics and Automation ({ICRA})",
%   year="2014",
%   month="May-June",
%   address="Hong Kong, China",
%   pages="3339--3344"
% }
%
% @article{Paige81,
%   author = "Paige, C. and Van Loan, C.",
%   title = "A {S}chur decomposition for {H}amiltonian matrices",
%   journal = "Linear Algebra and its Applications",
%   volume = "41",
%   pages = "11--32",
%   year = "1981",
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Danilo Bruno (danilo.bruno@iit.it), Sylvain Calinon (http://calinon.ch/)
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
Z = [A -G; -Q -A']; 

[U0, S0] = schur(Z); 
U = ordschur(U0, S0, 'lhp'); %Not yet available in GNU Octave 
X = U(n+1:end,1:n) / U(1:n,1:n); 
