function hh=streakarrow(X0,Y0,U,V,np,arrow)

%H = STREAKARROW(X,Y,U,V,np,arrow) creates "curved" vectors, from 
% 2D vector data U and V. All vectors have the same length. The
% magnitude of the vector is color coded.
% The arrays X and Y defines the coordinates for U and V.
% The variable np is a coefficient >0, changing the length of the vectors.
%     np=1 corresponds to a whole meshgrid step. np>1 allows ovelaps like
%     streamlines.
% The parameter arrow defines the type of plot: 
%   arrow=1 draws "curved" vectors
%   arrow=0 draws circle markers with streaks like "tuft" in wind tunnel
%   studies

% Example:
    %load wind
    %N=5; X0=x(:,:,N); Y0=y(:,:,N); U=u(:,:,N); V=v(:,:,N);
    %H=streakarrow(X0,Y0,U,V,1.5,0); box on; 
    
% Bertrand Dano 10-25-08
% Copyright 1984-2008 The MathWorks, Inc. 

colTmp=[0 0 0];
    
DX=abs(X0(1,1)-X0(1,2)); DY=abs(Y0(1,1)-Y0(2,1)); DD=min([DX DY]);
ks=DD/100;      % Size of the "dot" for the tuft graphs
np=np*10;   
alpha = 1.8;  % Size of arrow head relative to the length of the vector
beta = .22; % Width of the base of the arrow head relative to the length

XY=stream2(X0,Y0,U,V,X0,Y0);
%np=15;
Vmag=sqrt(U.^2+V.^2);
Vmin=min(Vmag(:)); Vmax=max(Vmag(:));
Vmag=Vmag(:); x0=X0(:); y0=Y0(:);

%ks=.1;
cmap=colormap;
for k=1:length(XY)
  F=XY(k); [L M]=size(F{1});
  if L<np
    F0{1}=F{1}(1:L,:);
    if L==1
      F1{1}=F{1}(L,:);
    else
      F1{1}=F{1}(L-1:L,:);
    end

  else
    F0{1}=F{1}(1:np,:);
    F1{1}=F{1}(np-1:np,:);
  end
  P=F1{1};
  vcol=floor((Vmag(k)-Vmin)./(Vmax-Vmin)*64); if vcol==0; vcol=1; end
  COL=[cmap(vcol,1) cmap(vcol,2) cmap(vcol,3)];
  hh=streamline(F0);
  %set(hh,'color',COL,'linewidth',.5);
  set(hh,'color',colTmp,'linewidth',1.5);

  if arrow==1&L>1
    x1=P(1,1); y1=P(1,2); x2=P(2,1); y2=P(2,2);
    u=x1-x2; v=y1-y2; u=-u; v=-v;
    %SC
    x2=x2+0.8*u;
    y2=y2+0.8*v;
    xa1=x2+u-alpha*(u+beta*(v+eps)); xa2=x2+u-alpha*(u-beta*(v+eps));
    ya1=y2+v-alpha*(v-beta*(u+eps)); ya2=y2+v-alpha*(v+beta*(u+eps));
    patch([xa1 x2 xa2 xa1],[ya1 y2 ya2 ya1],colTmp,'edgecolor',colTmp); hold on;
  else
    rectangle('position',[x0(k)-ks/2 y0(k)-ks/2 ks ks],'curvature',[1 1],'facecolor',colTmp, 'edgecolor',colTmp);
  end          
end

%axis image

%colorbar vert
%h=colorbar; 
%set(h,'ylim',[Vmin Vmax])

