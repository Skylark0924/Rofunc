function d = KLdiv(mn0, sig0, mn1, sig1)
% D-KL( N(mn0, sig0) || N(mn1, sig1) )
d = .5 * (trace(sig1\sig0) + (mn1-mn0)'/sig1*(mn1-mn0) - length(mn0) + log(det(sig1)/det(sig0)));