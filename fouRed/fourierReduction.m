function [Ipsf, d12, Mask] = fourierReduction(Gw, A, At, imsize, param)
% Flags

% Flag to pull up the values of elements of the holographic matrix
% This is to avoid having VERY small values which might later explode
% during computation of inverse or reciprocal.
if ~isfield(param, 'klargestpercent') 
    param.klargestpercent = 100; 
end

if ~isfield(param, 'diagthresholdepsilon') 
    param.diagthresholdepsilon = 1e-10; 
end

if ~isfield(param, 'covmatfileexists')
    param.covmatfileexists = 0;
end

if ~isfield(param, 'covmatfile')
    param.covmatfile = 'covariancemat.mat';
end

if ~isfield(param, 'fastCov')
    param.fastCov = 0;
end

klargestpercent = param.klargestpercent;
diagthresholdepsilon = param.diagthresholdepsilon;
covmatfileexists = param.covmatfileexists;
covmatfile = param.covmatfile;
fastCov = param.fastCov;


% Flag to load from a previously saved covariance matrix file
% covmatfileexists = 0;       % Read precomputed matrix 
% covmatfile = 'data/savedfiles/covariancemat.mat';

% Flag to set if we want to approximate D with an
% identity matrix. Reset the flag to use the normal D.
% (D is the diagonal aproximation of the covariance matrix)
Ny = imsize(1);
Nx = imsize(2);
% Compute holographic matrix
H = Gw'*Gw;

% Create the new measurement operator
serialise = @(x) x(:);

Ipsf = @(x) At(H*A(reshape(x, Ny, Nx)));  % Phi^T Phi = At G' G A = At H A: vect -> vect

% R = @(x) serialise(fft2(real(At(Gw'*x(:)))));   % F Phi^T: vect -> vect
% Rt = @(x) Gw*serialise(A(real(ifft2(reshape(full(x), Ny, Nx)))));   % Phi F^T: vect -> vect
% R = @(x) serialise(fftshift(fft2(ifftshift(At(Gw'*x(:))))));   % F Phi^T: vect -> vect
% Rt = @(x) Gw*serialise(A(fftshift(ifft2(ifftshift(reshape(full(x), Ny, Nx))))));   % Phi F^T: vect -> vect

fprintf('\nComputing covariance matrix...');
% Takes a vectorized input
%     covoperator = @(x) serialise(fft2(grid2img_fwd((Ny*Nx)*ifft2(reshape(full(x), [Ny, Nx])))));
% covoperator = @(x) R(Rt(x));
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
    
% d = ones(size(d)); % Disable weighting, simply do FPhi^TPhiF^T. Throw away the diagonal of covariancemat
fprintf('\nPruning covariancemat according to eigenvalues (diagonal elements)...\n');
Mask = (d >= prctile(d,100-klargestpercent));
d = d(Mask);
d = max(diagthresholdepsilon, d);  % This ensures that inverting the values will not explode in computation
d12 = 1./sqrt(d);
% Mask = sparse(1:length(nonzerocols), nonzerocols, ones(length(nonzerocols), 1), length(nonzerocols), (Ny*Nx));
end
