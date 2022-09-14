function plotMatrix(M)

figure,


[r, c] = size(M);



mx = max(max(M));
mn = min(min(M));

Md = round((M - mn)/(mx-mn)*size(gray, 1));

image([1 r], [1 c], Md);
colormap(1-gray)
% h = colorbar;
% set(h, 'YTick', (mx-mn)*get(h, 'YTick')./ (get(h, 'Ylim')*[0; 1]))
% set(h, 'Ylim', [mn, mx])
% 
% keyboard

title(['Min: ', num2str(mn), '     Max: ', num2str(mx)])