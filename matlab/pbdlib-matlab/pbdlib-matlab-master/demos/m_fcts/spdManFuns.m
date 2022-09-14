classdef spdManFuns
%%% separate class for functions on the SPD manifold based on Mixest
%%% spdfactory
%%% Ajay Tanwani, 2016

    methods (Static)
		%%
		
		function ManName = Name()
			ManName = 'SPD';
		end
		
		%% Inner product on SPD manifold of tangent vectors U and V about X
		
		function result = Inner(X, U, V) 
			
			result = real(sum(sum( (X\U).' .* (X\V) ))); %U(:).'*V(:);

		end
        %% norm of tangent vector U about X
		
		function result = Norm(X, U)
			
			result = sqrt(real(sum(sum( abs(X\U).^2 ))));
		end
		
		%% spd manifold distance between two points
		function d = Dist(X,Y)
			
			d = eig(X, Y);
			d = norm(log(d));
			
		end
		
		%% projection/tangent space of hermitian matrix U
		function Up = Proj(X, U)
			% Tangent space of symitian matrices is also a symitian matrix
			Up = sym(U);
		end

		%% conversion of Euclidean gradient to Riemannian gradient
		
		function Up = egrad2regrad(X, U)
			Up = X * sym(U) * X;
		end

		%% conversion of Euclidean Hessian to Riemannian hessian
		
		function Hess = ehess2rhess(X, egrad, ehess, eta)
			Hess = X*sym(ehess)*X + 2*sym(H*sym(egrad)*X);
			Hess = Hess - sym(eta*sym(egrad)*X);
		end
		
		%% retraction of U to the manifold about X
		
		function Y = retraction(X, U, t)
			if nargin < 3
				t = 1.0;
			end
% 			if flag
				E = t*U;
				Y = X * expm(X\E);
				Y = sym(Y);
% 			else
% 				Y = X + t*U;
% 			end
		end
		
		%% exponential map of U about X
		
		function Y = exponential(X, U, t)
			if nargin == 2
				t = 1;
			end
			Y = retraction(X, U, t);
		end
		
		%% logarithmic map of Y about X		
		function U = logarithm(X, Y)
			U = X*logm(X\Y);
			U = sym(U);
		end
		
		%% generate a random sample of covariance matrix
		function X = randomSample(n)
			X = randn(n);
			X = (X*X');
		end
		
		%% generate a random vector on SPD manifold
		function U = randomVec(n)
			U = randn(n);
			U = sym(U);
			U = U / norm(U,'fro');
		end
		
		%% Parallel Transport of E about X to Y
		function F = transpvec(X, Y, E)
			expconstruct= sqrtm(Y/X);
			F = expconstruct*E*expconstruct';
		end
		
		%% parallel transport bundle of tangents defined on X to Y
		function [expconstruct,iexpconstruct] = transpvecf(X, Y)
			expconstruct= sqrtm(Y/X);
			%F = expconstruct*E*expconstruct';
			if nargout > 1
				iexpconstruct = inv(expconstruct);
			end
		end
	
		%% inverse of vector transport   
		function F = itranspvec(X, Y, E)
			F = transpvec(Y, X, E);
		end
		
		%% faster version of vector transport by storing some information
		
		function F = transpvecfast(expconstruct, E)			
			F = expconstruct*E*expconstruct';
		end
		
		%% faster version of inverse vector transport by storing some information
		
		function F = itranspvecfast(iexpconstruct, E)			
			F = iexpconstruct*E*iexpconstruct';
		end
		
		%% Linear combination of tangent vectors
		
		function d = lincomb(x, a1, d1, a2, d2)
			
			if nargin == 3
				d = a1*d1;
			elseif nargin == 5
				d = a1*d1 + a2*d2;
			else
				error('Bad use of psd.lincomb.');
			end			
		end
		
	end
		
end

%% yied symmetric matrix

function result = sym(U)

	result = (U+U')/2;
	
end
	