classdef simplexManFuns
%%% separate class for functions on the SPD manifold based on Mixest
%%% spdfactory
%%% Ajay Tanwani, 2016

    methods (Static)
		%%
		
		function ManName = Name()
			ManName = 'simplex';
		end
		
		%% Inner product on simplex manifold of tangent vectors U and V about X
		
		function result = Inner(x, d1, d2) 
			
			result = d1(:).'*d2(:);

		end
        %% norm of vector in simplex space
		
		function result = Norm(x, d)
			
			result = norm(d, 'fro');
		end
		
		%% simplex manifold distance between two points
		function dist = Dist(x,y)
			
			dist = norm(x-y, 'fro');
			
		end
		
		%% projection of vector on simplex is the vector itself
		function Up = Proj(x, d)
			% Tangent space of symitian matrices is also a symitian matrix
			Up = d;
		end

		%% conversion of Euclidean gradient to Riemannian gradient
		
		function gn = egrad2rgrad(x, g)
			Cov = diag(x) - x * x';
			gn = Cov(1:end-1,:)*g;
		end

		%% conversion of Euclidean Hessian to Riemannian hessian
		
% 		function Hess = ehess2rhess(x, egrad, ehess, eta)
% 		end
		
		%% retraction of vector to the manifold is same as exponential map
		
		function y = retraction(x, d, t)
			y = simplexManFuns.expmap(x,d,t);
		end
		
		%% exponential map of U about X
		
		function y = expmap(x, d, t)
			
			if length(x) == 1
				y = x;
				return;
			end
			n = size(x,1);
			
			% apply first the following change of variable
			xn = log(x(1:n-1)) - log(x(n));
			
			% moving in the variable change domain
			if nargin == 3
				yn = xn + t*d;
			else
				yn = xn + d;
			end
			
			% going back to the original domain
			y = exp([yn;0]);
			y = y / sum(y);
		end
		
		%% logarithmic map of Y about X
		
		function d = logmap(x, y)
			if length(x) == 1
				d = 0;
				return;
			end
			n = size(y,1);
			
			% apply first the following change of variable
			xn = log(x(1:n-1)) - log(x(n));
			yn = log(y(1:n-1)) - log(y(n));
			
			% distance between transformed points
			d = yn - xn;
		end
		
		%% generate a random sample of covariance matrix
		function x = random()
			x = rand(n,1);
			x = x / sum(x);
		end
		
		%% generate a random vector on SPD manifold
		function u = randvec(x) %#ok<INUSD>
			u = randn(n-1,1);
			u = u / norm(u, 'fro');
		end
		
		%% Parallel Transport of d from x1 to x2
		
		function F = transp(x1, x2, d)
			F = d;
		end
		
		%% Linear combination of tangent vectors
		
		function v = lincomb(x, a1, d1, a2, d2) %#ok<INUSL>
			if nargin == 3
				v = a1*d1;
			elseif nargin == 5
				v = a1*d1 + a2*d2;
			else
				error('Bad usage of simplex.lincomb');
			end
		end
		
	end
		
end

%% yied symmetric matrix

function result = sym(U)

	result = (U+U')/2;
	
end
	