function [Yhat, B1, B2, alpha] = matrixRR(XTr, YTr, XTe, rank, lambda, maxDiffCriterion, maxIter)
% Predict a new output with Ridge regression for matrix data
% Based on [Gua, Kotsia, Patras. 2012] and [Zhou, Li, Zhu. 2013]
% https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2013.776499
%
% XTr		Input N x D1 x D2
% YTr		Output N x d
% XTe		New input Nte x D1 x D2
% lambda	RR parameter
% Yhat		Predicted output Nte x d
% B1        Left multiplying matrix
% B2        Right multiplying matrix
% alpha     Constant term
%
% Noémie Jaquier, 2018.

if nargin < 6
    maxDiffCriterion = 1e-4;
end
if nargin < 7
    maxIter = 100;
end

% Data dimensions
[N, d1, d2] = size(XTr);
[N, d] = size(YTr);

% Initialization
B1 = randn(d1, rank);
B2 = randn(d2, rank);
alpha = randn(d,1);

% Permute XTr
XTr = permute(XTr, [2,3,1]);

% Optimization of parameters
prevRes = 0;
for it = 1:maxIter
    % B1
    for n = 1:N
        Ztmp = XTr(:,:,n)*B2;
        zVec1(n,:) = Ztmp(:);
    end
    b1Tmp = (zVec1'*zVec1 + lambda*eye(d1*rank)) \ zVec1'*(YTr - alpha);
    B1 = reshape(b1Tmp, d1, rank);
    
    % B2
    for n = 1:N
        Ztmp = XTr(:,:,n)'*B1;
        zVec2(n,:) = Ztmp(:);
    end
    b2Tmp = (zVec2'*zVec2 + lambda*eye(d2*rank)) \ zVec2'*(YTr - alpha);
    B2 = reshape(b2Tmp, d2, rank);
    
    % alpha
    Bvec = khatriRao(B2,B1)*ones(rank,1);
    alpha = 0;
    for n = 1:N
        xtmp = XTr(:,:,n);
        alpha = alpha + YTr(n,:) - Bvec'*xtmp(:);
    end
    alpha = alpha/N;
    
    % Compute residuals
    res = 0;
    for n=1:N
        xtmp = XTr(:,:,n);
        res = res + (YTr(n,:) - alpha - Bvec'*xtmp(:))^2;
    end
    
    % Check convergence
    resDiff = prevRes - res;
    if resDiff < maxDiffCriterion && it > 1
        % Prediction
        Nte = size(XTe, 1);
        for n = 1:Nte
            xtmp = permute(XTe(n,:,:),[2,3,1]);
            Yhat(n,:) = alpha + Bvec'*xtmp(:);
        end
        disp(['Matrix RR converged after ' num2str(it) ' iterations.']);
        return
    end
    prevRes = res;
end

% Prediction
Nte = size(XTe, 1);
for n = 1:Nte
    xtmp = permute(XTe(n,:,:),[2,3,1]);
    Yhat(n,:) = alpha + Bvec'*xtmp(:);
end
disp(['Matrix RR did not converge (res diff ' num2str(resDiff) ').']);
end