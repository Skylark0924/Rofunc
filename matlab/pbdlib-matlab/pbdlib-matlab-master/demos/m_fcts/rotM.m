function rot = rotM(v,angle)
% ROTM : 3 x 3 rotation matrix 
%           [rot]= rotM(v) gives the rotation that moves v to the north
%                                           pole
%           [rot]= rotM(v, angle) gives the rotation with axis 'v' and
%           angle 'angle'
%
%          Use: 3x3 matrix rot can be pre-multiplied to 3 x n matrix 'data'
%               to have a rotated data set.
%
%           % Example: generate data on a unit sphere
%           n = 50;
%           theta = linspace(0,pi*1.5,n);
%           data = 5*[cos(theta); sin(theta)] + randn(2,n);
%           data = data/10;
%           data = ExpNP(data);
%           % calculate extrinsic mean                
%           c0 = mean(data,2);
%           c0 = c0/norm(c0);
%           % rotate whole data in a way that the EM moves to the north
%           % pole.
%           rotM(c0)*data
%
%
%   See also ExpNP, LogNP, geodmeanS2.

% Last updated Aug 10, 2009
% Sungkyu Jung


if nargin == 1 % then perform rotation that moves 'v' to the 'north pole'.
   st      = v / norm(v);
   acangle = st(3);
   cosa    = acangle;
   sina    = sqrt(1-acangle^2);
   if (1-acangle)>1e-16
      v = [st(2);-st(1);0]/sina;
   else
      v = [0;0;0];
   end   
else    % then perform rotation with axis 'v' and angle 'angle'
    v = v/norm(v);
    cosa = cos(angle);
    sina = sin(angle);
end

vera = 1 - cosa;

x = v(1);
y = v(2);
z = v(3);

rot = [cosa+x^2*vera x*y*vera-z*sina x*z*vera+y*sina; ...
       x*y*vera+z*sina cosa+y^2*vera y*z*vera-x*sina; ...
       x*z*vera-y*sina y*z*vera+x*sina cosa+z^2*vera];
