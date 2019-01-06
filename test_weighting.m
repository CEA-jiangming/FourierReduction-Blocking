% This file is to compare two approches to compute covariance matrix for
% dimensionality reduction in different scenarios
% Approch 1: F * Ipsf * Dirac2D, gridded natural weights in practice
% Approch 2: matrix probing, very very slow


close all
clear variables
clc

addpath data
addpath data/images
addpath lib/
addpath fouRed/
% addpath src
addpath irt/

try
    setup;
catch ME
    error('NUFFT library not found in location src/irt');
end

rng('shuffle');

Nx = 64;
Ny = 64;

serialise = @(x) x(:);
%% Gridded visibilities
ratio = 0.1;

M = round(Nx*Ny*ratio);

u = [];
v = [];

ox = 1;
oy = 1;

sigma = 5;

while length(u) < M
    u = round(sigma * randn(M, 1) + Nx/2);
    v = round(sigma * randn(M, 1) + Ny/2);
    
    sfu = find((u<=Nx) & (u>=1));
    sfv = find((v<=Ny) & (v>=1));
    sf = intersect(sfu, sfv);
    
    u = u(sf);
    v = v(sf);
end

figure(), plot(u,v,'.'), title('Grided uv-coverage')

mu = [Nx/2*ox Ny/2*oy];
Sigma = [20*ox 0; 0 20*oy];
x1 = 1:(Nx*ox); x2 = 1:(Ny*oy);
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);

F = reshape(F,length(x2),length(x1));
F = 10*F/max(F(:));
figure(), surf(x1,x2,F), title('Simulated weighting')

W = zeros(ox*Nx, oy*Ny);
W(ox*u,oy*v) = F(ox*u,oy*v);
figure(), surf(x1,x2,W), title('Simulated weighted sampling')

Phi = @(x) serialise(W.*fftshift(fft2(ifftshift(reshape(full(x), Nx, Ny)), ox*Nx, oy*Ny)));
Phit = @(x) serialise(fftshift(ifft2(ifftshift(W'.*reshape(full(x), ox*Nx, oy*Ny)), Nx, Ny)));

Ipsf = @(x) Phit(Phi(x));

im = zeros(Nx, Ny);
im(ceil((Nx+1)/2), ceil((Ny+1)/2)) = 1;

PSF = reshape(Ipsf(im), Nx, Ny);
FPSF = fftshift(fft2(ifftshift(PSF)));

figure(),imagesc(abs(FPSF)),colorbar(), title('Sigma matrix via PSF')

covoperator = @(x) fftshift(fft2(ifftshift(reshape(Phit(Phi(fftshift(ifft2(ifftshift(reshape(full(x), Nx, Ny)))))), Nx, Ny))));

covariancemat = guessmatrix_test2(1, covoperator, Ny*Nx, Ny*Nx);
d = diag(covariancemat);

figure(),imagesc(reshape(abs(d), Nx, Ny)),colorbar(), title('Sigma matrix via matrix probing')

diff = FPSF - reshape(d, Nx, Ny);
figure(),imagesc(abs(diff)),colorbar(), title('Different between two methods (gridded vis)')

%% Continuous visibilities

N = Nx * Ny;
visibSize = 5 * Ny * Nx;

ox = 2; % oversampling factors for nufft
oy = 2; % oversampling factors for nufft
Kx = 8; % number of neighbours for nufft
Ky = 8; % number of neighbours for nufft

sampling_pattern = 'gaussian';

util_gen_sampling_pattern_config; % Set all parameters
sparam.N = N; % number of pixels in the image
sparam.Nox = ox*Nx; % number of pixels in the image
sparam.Noy = oy*Ny; % number of pixels in the image
sparam.p = ceil(visibSize/N);

[~, ~, uw, vw, ~] = util_gen_sampling_pattern(sampling_pattern, sparam);
uw = [uw; -uw];
vw = [vw; -vw];

% uw = linspace(-pi, pi, Nx);
% vw = linspace(-pi, pi, Ny);
% uw = uw(u);
% vw = vw(v);
% uw = uw(:);
% vw = vw(:);

% u = linspace(-pi, pi, Nx);
% v = linspace(-pi, pi, Ny);
% [uw,vw] = meshgrid(u,v);
% uw = uw(:);
% vw = vw(:);

figure(),plot(uw,vw,'.'),title('continuous uv-coverage')

fprintf('Initializing the NUFFT operator\n\n');
tstart = tic;
[A, At, Gw, scale] = op_nufft([uw vw], [Nx Ny], [Kx Ky], [oy*Nx ox*Ny], [Nx/2 Ny/2], 0);
tend = toc(tstart);
fprintf('Initialization runtime: %ds\n\n', ceil(tend));

Phi = @(x) Gw*serialise(A(reshape(full(x), Nx, Ny)));   % Phi: vect -> vect
Phit = @(x) serialise((At(Gw'*x(:))));

Ipsf = @(x) Phit(Phi(x));

dirac2D = zeros(Nx, Ny);
dirac2D(ceil((Nx+1)/2), ceil((Ny+1)/2)) = 1;

PSF = reshape(Ipsf(dirac2D), Nx, Ny);
PSF1 = zeros(size(PSF));
% PSF1(ceil((Nx+1)/2)-8:ceil((Nx+1)/2)+8, ceil((Ny+1)/2)-8:ceil((Ny+1)/2)+8) = PSF(ceil((Nx+1)/2)-8:ceil((Nx+1)/2)+8, ceil((Ny+1)/2)-8:ceil((Ny+1)/2)+8);
FPSF = fftshift(fft2(ifftshift(PSF)));

figure(),imagesc(abs(FPSF)),colorbar(), title('Sigma matrix via PSF')
covoperator = @(x) fftshift(fft2(ifftshift(reshape(Phit(Phi(fftshift(ifft2(ifftshift(reshape(full(x), Nx, Ny)))))), Nx, Ny))));

covariancemat = guessmatrix_test2(1, covoperator, Ny*Nx, Ny*Nx);
d = diag(covariancemat);

figure(),imagesc(reshape(abs(d), Nx, Ny)),colorbar(), title('Sigma matrix via matrix probing')

diff = FPSF - reshape(d, Nx, Ny);
figure(),imagesc(abs(diff)),colorbar(), title('Different between two methods (continuous vis)')

%% Phi^T Phi matrix probing
figure(),imagesc(abs(PSF)),colorbar(), title('Phi^T Phi via PSF')

covariancemat2 = guessmatrix_test2(1, Ipsf, Ny*Nx, Ny*Nx);
d2 = diag(covariancemat2);
% d3 = fftshift(fft2(ifftshift(reshape(full(d2), Nx, Ny))));

figure(),imagesc(reshape(abs(d2), Nx, Ny)),colorbar(), title('Phi^T Phi probing via matrix probing')

diff2 = PSF - reshape(d2, Nx, Ny);
figure(),imagesc(abs(diff2)),colorbar(), title('Different between two methods')









