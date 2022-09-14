% GRAYLICEXTERNAL is an internal command of the toolbox. It uses Fast LIC method
% implemented as an external Matlab function to generate an intensity image.
% Usage:
% [LICIMAGE, INTENSITY,NORMVX,NORMVY] = GRAYLICEXTERNAL(VX, VY, ITERATIONS);
% VX and VY should contain X and Y components of the vector field. They
% should be M x N floating point arrays with equal sizes.
%
% ITERATIONS is an integer number for the number of iterations used in
% Iterative LIC method. use number 2 or 3 to get a more coherent output
% image.
%
% LICIMAGE returns an M x N floating point array containing LIC intensity image 
% INTENSITY returns an M x N floating point array containing magnitude of vector in the field 
% NORMVX and NORMVY contain normalized (each vector is normalized to have
% the length of 1.0) components of the vector field

function [LICImage, intensity,normvx,normvy] = grayLICExternal(vx,vy, iterations)
[width,height] = size(vx);
LIClength = round(max([width,height]) / 30); 

kernel = ones(2 * LIClength); 
%kernel = ones(7);

LICImage = zeros(width, height);
intensity = ones(width, height); % array containing vector intensity

% Making white noise
rand('state',0) % reset random generator to original state
noiseImage= rand(width,height);
% for i=1:width
%   for j=1:height
%     if noiseImage(i,j)<0.9
%       noiseImage(i,j)=0;
%     end
%   end
% end

%noiseImage=double(imread('imgbw03.png'))./255;

% Making LIC Image
for m = 1:iterations
[LICImage, intensity,normvx,normvy] = fastLICFunction(vx,vy,noiseImage,kernel); % External Fast LIC implemennted in C language
%[LICImage, intensity,normvx,normvy] = grayFastLIC2(vx,vy,iterations,noiseImage);
%[LICImage, intensity,normvx,normvy] = grayLIC2(vx,vy,iterations,noiseImage);

LICImage = imadjust(LICImage).*0.5 + 0.5; % Adjust the value range
noiseImage = LICImage; 
end;




