function [Ipsf, d12, Mask] = fourierReduction(Gw, A, At, imsize, param)

% Flags monitoring

if ~isfield(param,'enable_klargestpercent') && ~isfield(param,'enable_estimatethreshold')
    param.enable_klargestpercent = 1;
end

if param.enable_klargestpercent
    if ~isfield(param, 'klargestpercent') 
        param.klargestpercent = 100; 
    end
    klargestpercent = param.klargestpercent;
elseif param.enable_estimatethreshold
    if ~isfield(param, 'gamma') 
        param.gamma = 3; 
    end
    if ~isfield(param, 'x2') || ~isfield(param, 'noise')
        error('Either ||x||_2 or noise is missing for the estimation of the threshold');
    end
    noise = param.noise;
    gamma = param.gamma;
    x2 = param.x2;
end

% Flag to pull up the values of elements of the holographic matrix
% This is to avoid having VERY small values which might later explode
% during computation of inverse or reciprocal.
if ~isfield(param, 'diagthresholdepsilon') 
    param.diagthresholdepsilon = 1e-10; 
end

if ~isfield(param, 'covmatfileexists')
    param.covmatfileexists = 0;
end

if ~isfield(param, 'covmatfile')
    param.covmatfile = 'covariancemat.mat';
end

% Fast covariance matrix computation
if ~isfield(param, 'fastCov')
    param.fastCov = 1;
end

diagthresholdepsilon = param.diagthresholdepsilon;
covmatfileexists = param.covmatfileexists;
covmatfile = param.covmatfile;
fastCov = param.fastCov;

Ny = imsize(1);
Nx = imsize(2);
% Compute holographic matrix
H = Gw'*Gw;

% Create the new measurement operator
serialise = @(x) x(:);

Ipsf = @(x) At(H*A(reshape(x, Ny, Nx)));  % Phi^T Phi = At G' G A = At H A: vect -> vect

fprintf('\nComputing covariance matrix...');
% Covariance operator F Phi^T Phi F^T = F At H A F^T = F Ipsf F^T
covoperator = @(x) serialise(fftshift(fft2(ifftshift(Ipsf(fftshift(ifft2(ifftshift(reshape(full(x), Ny, Nx)))))))));
diagonly = 1; % Only compute diagonal of covariance matrix FPhi^TPhiF^T
if covmatfileexists
    fprintf('\nLoading covariance matrix from file...');
    load(covmatfile, 'covariancemat');
else
    tstartcovmat = tic;
    if fastCov
        dirac2D = zeros(Ny, Nx);
        dirac2D(ceil((Ny+1)/2), ceil((Nx+1)/2)) = 1;

        PSF = reshape(Ipsf(dirac2D), Ny, Nx);
        covariancemat = fftshift(fft2(ifftshift(PSF)));
    else
        covariancemat = guessmatrix(diagonly, covoperator, Ny*Nx, Ny*Nx);
    end
    fprintf('\nSaving covariance matrix...\n');
    save(covmatfile, 'covariancemat');
    tendcovmat = toc(tstartcovmat);
    fprintf('Time to compute covariance matrix: %e s\n', tendcovmat)
end

if fastCov
    d = abs(covariancemat(:));
else
    d = diag(covariancemat); %*(sigma_noise^2)
    d = abs(d);
end

% Singular values thresholding
fprintf('\nPruning covariancemat according to eigenvalues (diagonal elements)...\n');
if param.enable_klargestpercent
    Mask = (d >= prctile(d,100-klargestpercent));
elseif param.enable_estimatethreshold
    % Embed the noise
    rn = fftshift(fft2(ifftshift(At(Gw'*noise))));  % Apply F Phi
    th = gamma * std(rn(:)) / x2;
    Mask = (d >= th);
end
d = d(Mask);
fprintf('\nThe threshold is %e \n', min(d));

d = max(diagthresholdepsilon, d);  % This ensures that inverting the values will not explode in computation
d12 = 1./sqrt(d);

end
