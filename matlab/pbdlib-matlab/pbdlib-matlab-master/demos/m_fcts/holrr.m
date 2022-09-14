function W = holrr(X, Y, rank, gamma)
% Noemie Jaquier

d0 = size(X,2);

U = cell(1,length(rank));
Ut = cell(1,length(rank));

YTmp = tens2mat(Y,1);
UTmp = (X'*X + gamma.*eye(d0)) \ (X' * (YTmp * YTmp') * X);
[V,D] = eig(UTmp);
[~, ind] = sort(diag(D),'descend');
U{1,1} = V(:,ind(1:rank(1)));

for i = 2:length(rank)
	YTmp = tens2mat(Y,i);
	[V,D] = eig(YTmp*YTmp');
	[~, ind] = sort(diag(D),'descend');
	U{1,i} = V(:,ind(1:rank(i)));
	Ut{1,i} = U{1,i}';
end

Ut{1,1} = (U{1,1}' * (X'*X + gamma.*eye(d0)) * U{1,1}) \ (U{1,1}'*X');

G = tmprod(Y,Ut,1:length(rank));
W = tmprod(G,U,1:length(rank));

end
