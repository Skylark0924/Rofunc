function prob = logGaussPDF(Data, Mu, Sigma)
%PDF of univariate lognormal distribution
%Sylvain Calinon, 2015

[nbVar,nbData] = size(Data);
DataTmp = log(Data)' - repmat(Mu',nbData,1); %Not the same as gaussPDF!
prob = sum((DataTmp/Sigma).*DataTmp, 2);

% %Univariate
% prob = exp(-0.5*prob) ./ (Data*sqrt((2*pi)^nbVar*abs(det(Sigma)))+realmin)'; %Not the same as gaussPDF!

%Multivariate?
prob = exp(-0.5*prob) ./ (prod(Data,1)*sqrt((2*pi)^nbVar*abs(det(Sigma)))+realmin)'; %Not the same as gaussPDF!
%prob = exp(-0.5*prob) ./ (sum(Data.^2,1).^.5*sqrt((2*pi)^nbVar*abs(det(Sigma)))+realmin)'; %Not the same as gaussPDF!

%prob = (1./(Data*sqrt(2*pi*Sigma))) .* exp(-0.5*(log(Data)-Mu).^2/Sigma); %1D case
