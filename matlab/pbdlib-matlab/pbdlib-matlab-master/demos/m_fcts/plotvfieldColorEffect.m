function img = plotvfieldColorEffect(map)

colx = linspace(-pi,pi,7);  
coly = [.9 .9 .1 .1 .1 .9 .9; ...
        .1 .9 .9 .9 .1 .1 .1; ...
        .1 .1 .1 .9 .9 .9 .1];
			
[wy,wx] = size(map.vx);

nbIter = 3; %5 SCSCSCSC
grayImage = grayLICExternal(map.vy,map.vx,nbIter);

colAng = spline(colx,coly,map.ang);

%Add color depending on direction
for i=1:wx
  for j=1:wy
    %img(i,j,:) = [grayImage(i,j); grayImage(i,j); grayImage(i,j)];
    %img(i,j,:) = [grayImage(i,j); grayImage(i,j); grayImage(i,j)] + colAng(:,i,j);
    %img(i,j,:) = [grayImage(i,j); grayImage(i,j); grayImage(i,j)] .* min(map.Wmap(i,j)*1.5+.3,1);
    img(i,j,:) = ([grayImage(i,j); grayImage(i,j); grayImage(i,j)] + colAng(:,i,j)) .* min(map.Wmap(i,j)*1.5+.3,1);
  end
end

img = img ./ max(max(max(img)));
