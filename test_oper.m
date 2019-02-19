% This file is to compare gridded operator and complete operator
%
% The complete operator case:
% \Sigma^-1 F \Phi^T y = \Sigma^-1 F \Phi^T \Phi x + \Sigma^-1 F \Phi^T n
%
% The gridded operator case:
% F \Phi^T y = \Sigma^2 F x + F \Phi^T n
%

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

normalize_data = 0;
usingPrecondition = 0;
input_snr = 40;
num_tests = 1;
verbosity = 1;

serialise = @(x) x(:);

%% Read image and generate data
image_file_name = './data/images/M31_256.fits';

[im, N, Ny, Nx] = util_read_image(image_file_name);

ox = 2; % oversampling factors for nufft
oy = 2; % oversampling factors for nufft
Kx = 8; % number of neighbours for nufft
Ky = 8; % number of neighbours for nufft

visibSize = 2*Nx*Ny;
param_sampling.N = N; % number of pixels in the image
param_sampling.Nox = ox*Nx; % number of pixels in the image
param_sampling.Noy = oy*Ny; % number of pixels in the image
util_gen_sampling_pattern_config; % Set all parameters

[~, ~, uw, vw, ~] = util_gen_sampling_pattern(sampling_pattern, param_sampling);
uw = [uw; -uw];
vw = [vw; -vw];

%% measurement operator initialization
fprintf('Initializing the NUFFT operator\n\n');
tstart = tic;
[A, At, Gw, scale] = op_nufft([vw uw], [Ny Nx], [Ky Kx], [oy*Ny ox*Nx], [Ny/2 Nx/2], 0);
tend = toc(tstart);
fprintf('Initialization runtime: %ds\n\n', ceil(tend));

% use the absolute values to speed up the search
Gw_a = abs(Gw);

b_l = length(uw);
% check if eack line is entirely zero
W = Gw_a' * ones(b_l, 1) ~= 0;

% store only what we need from G
G = Gw(:, W);    
% end

%% sparsity operator definition
wlt_basis = {'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'self'}; % wavelet basis to be used
nlevel = 4; % wavelet level

[Psi, Psit] = op_p_sp_wlt_basis(wlt_basis, nlevel, Ny, Nx);
[Psiw, Psitw] = op_sp_wlt_basis(wlt_basis, nlevel, Ny, Nx);

%% generate noisy input data
 
for k = 1:num_tests
    % cell structure to adapt to the solvers
    if normalize_data
        [y0{k}{1}, ~, y{k}{1}, ~, sigma_noise,~, noise{k}{1}] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    else
        [y0{k}{1}, y{k}{1}, ~, ~, sigma_noise, noise{k}{1}, ~] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    end        
end

%% definition for the stopping criterion
% options:
% l2_ball_definition -> 'sigma', 'chi-percentile', 'value'
% stopping_criterion -> 'sigma', 'chi-percentile', 'l2-ball-percentage', 'value'

l2_ball_definition = 'sigma';
stopping_criterion = 'sigma';

param_l2_ball.stop_eps_v = sqrt(2*visibSize); % set epsilon value BEFORE running this script
param_l2_ball.val_eps_v = 1.0*param_l2_ball.stop_eps_v;

param_l2_ball.sigma_ball = 2;
param_l2_ball.sigma_stop = 2;

param_l2_ball.chi_percentile_ball = 0.99;
param_l2_ball.chi_percentile_stop = 0.999;

param_l2_ball.l2_ball_percentage_stop = 1.0001;

use_same_stop_criterion = 1; % forces the distributed criterion to be scaled
                             % such that same norm is imposed as in the nondistributed setup

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For dimensionality reduction

% Fourier reduction parameters
param_fouRed.enable_klargestpercent = 1;
param_fouRed.klargestpercent = 25;
param_fouRed.enable_estimatethreshold = 0;
param_fouRed.gamma = 3;             % By using threshold estimation, the optimal theshold reads as gamma * sigma / ||x||_2
param_fouRed.diagthresholdepsilon = 1e-10; 
param_fouRed.covmatfileexists = 0;
param_fouRed.covmatfile = 'covariancemat.mat';
param_fouRed.fastCov = 1;

if normalize_data
    Gw = sqrt(2)/sigma_noise * Gw;      % Whitening G matrix (embed natural weighting in the measurement operator). In reality, this should be done by natural weighting!
end

if param_fouRed.enable_estimatethreshold
    param_fouRed.x2 = norm(im);
    param_fouRed.noise = noise{k}{1};
end

fprintf('\nDimensionality reduction...');
% psf operator Ipsf, singular value matrix Sigma, mask matrix (to reduce the dimension)
[Ipsf, Sigma, Mask] = fourierReduction(Gw, A, At, [Ny, Nx], param_fouRed);
% New measurement operator C, new reduced measurement operator B
[C, Ct, B, Bt] = oper_fourierReduction(Ipsf, Sigma, Mask, [Ny, Nx]);
fprintf('\nDimensionality reduction is finished');

% Embed the y using the same reduction
for k = 1:num_tests
    y_grid = fftshift(fft2(ifftshift(At(Gw'*y{k}{1}))));
    y_grid = y_grid(:);
    y_grid = y_grid(Mask);
    yTmat = Sigma.*y_grid;
    yT{k} = {yTmat};
end

%Bound for the L2 norm
fprintf('Computing epsilon bound... ');
tstart1=tic;      

% Embed the noise
for k = 1:num_tests
    % Apply F Phi
    n_grid = fftshift(fft2(ifftshift(At(Gw'*noise{k}{1}))));
    n_grid = n_grid(:);
    n_grid = n_grid(Mask);
    epsilonT{k}{1} = norm(Sigma .* n_grid);
    epsilonTs{k}{1} = 1.001*epsilonT{1}{1};
    epsilon{k} = norm(cell2mat(epsilonT{k}));
    epsilons{k} = 1.001*epsilon{k};     % data fidelity error * 1.001
end

    %%%%%%%%%%%%%%%
fprintf('Done\n');
tend1=toc(tstart1);
fprintf('Time: %e\n', tend1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Case 1: Complete operator %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T = {Sigma};
% W = {Mask};
% 
% evl = op_norm(B, Bt, [Ny, Nx], 1e-4, 200, verbosity);

%% PDFB parameter structure sent to the algorithm
param_pdfb.im = im; % original image, used to compute the SNR
param_pdfb.verbose = verbosity; % print log or not
param_pdfb.nu1 = 1; % bound on the norm of the operator Psi
% param_pdfb.nu2 = evl; % bound on the norm of the operator A*G
param_pdfb.gamma = 1e-6; % convergence parameter L1 (soft th parameter)
param_pdfb.tau = 0.49; % forward descent step size
param_pdfb.rel_obj = 1e-4; % stopping criterion
param_pdfb.max_iter = 500; % max number of iterations
param_pdfb.lambda0 = 1; % relaxation step for primal update
param_pdfb.lambda1 = 1; % relaxation step for L1 dual update
param_pdfb.lambda2 = 1; % relaxation step for L2 dual update
param_pdfb.sol_steps = [inf]; % saves images at the given iterations

param_pdfb.use_proj_elipse_fb = 1;
param_pdfb.elipse_proj_max_iter = 200;
param_pdfb.elipse_proj_min_iter = 1;
param_pdfb.elipse_proj_eps = 1e-8; % precision of the projection onto the ellipsoid

param_pdfb.use_reweight_steps = 4;
param_pdfb.use_reweight_eps = 0;
param_pdfb.reweight_steps = [600:50:10000 inf];
param_pdfb.reweight_rel_obj = 1e-5; % criterion for performing reweighting
param_pdfb.reweight_min_steps_rel_obj = 50;
param_pdfb.reweight_alpha = 1; % Alpha always 1
param_pdfb.reweight_alpha_ff = 0.75; % 0.25 Too agressively reduces the weights, try 0.7, 0.8
param_pdfb.reweight_abs_of_max = inf;
param_pdfb.total_reweights = 20;

param_pdfb.use_best_bound_steps = 0;
param_pdfb.use_best_bound_eps = 0;
param_pdfb.best_bound_reweight_steps = 0;
param_pdfb.best_bound_steps = [inf];
param_pdfb.best_bound_rel_obj = 1e-6;
param_pdfb.best_bound_alpha = 1.0001; % stop criterion over eps bound
param_pdfb.best_bound_alpha_ff = 0.998;
param_pdfb.best_bound_stop_eps_v = 1.001*param_l2_ball.stop_eps_v; % the method stops if the eps bound goes below this

param_pdfb.use_adapt_bound_eps = 0;
param_pdfb.adapt_bound_steps = 100;
param_pdfb.adapt_bound_rel_obj = 1e-5;
param_pdfb.hard_thres = 0;
param_pdfb.adapt_bound_tol =1e-3;
param_pdfb.adapt_bound_start = 1000;

result_st = [];
result_st.sol = cell(num_tests, 1);
result_st.L1_v = cell(num_tests, 1);
result_st.L1_vp = cell(num_tests, 1);
result_st.L2_v = cell(num_tests, 1);
result_st.L2_vp = cell(num_tests, 1);
result_st.time = cell(num_tests, 1);
result_st.delta_v = cell(num_tests, 1);
result_st.sol_v = cell(num_tests, 1);
result_st.sol_reweight_v = cell(num_tests, 1);
result_st.snr_v = cell(num_tests, 1);

result_st.snr = cell(num_tests, 1);
result_st.sparsity = cell(num_tests, 1);
result_st.no_itr = cell(num_tests, 1);

result_st.singkept = cell(num_tests, 1);


% for i = 1:num_tests
%     % wavelet mode is a global variable which does not get transfered
%     % to the workes; we need to set it manually for each worker
%     dwtmode('per');
% 
%     fprintf('Test run %i:\n', i);
% 
%     tstart_a = tic;
%     fprintf(' Running pdfb_bpcon_par_sim_rescaled\n');
%     [result_st.sol{i}, result_st.L1_v{i}, result_st.L1_vp{i}, result_st.L2_v{i}, ...
%         result_st.L2_vp{i}, result_st.delta_v{i}, result_st.sol_v{i}, result_st.snr_v{i}, ~, ~, result_st.sol_reweight_v{i}] ...
%         = pdfb_bpcon_par_sing_sim_rescaled_adapt_eps(yT{i}, [Ny, Nx], epsilonT{i}, epsilonTs{i}, epsilon{i}, epsilons{i}, C, Ct, T, W, Psi, Psit, Psiw, Psitw, param_pdfb);
%     tend = toc(tstart_a);
%     fprintf(' pdfb_bpcon_par_sing_sim_rescaled runtime: %ds\n\n', ceil(tend));
% 
%     result_st.time{i} = tend;
% 
%     result_st.singkept{i} = sum(W{i})/numel(W{i});
% 
%     error = im - result_st.sol{i};
%     result_st.snr{i} = 20 * log10(norm(im(:))/norm(error(:)));
%   
%     result_st.no_itr{i} = length(result_st.L1_v{i});
% 
%     wcoef = [];
%     for q = 1:length(Psit)
%         wcoef = [wcoef; Psit{q}(result_st.sol{i})];
%     end
%     result_st.sparsity{i} = sum(abs(wcoef) > 1e-3)/length(wcoef);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Case 2: Gidded operator %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear T W;
% Fx =@(x) serialise(fftshift(fft2(ifftshift(reshape(x, Ny, Nx)))));
% Fxt =@(x) fftshift(fft2(ifftshift(reshape(x, Ny, Nx))));
% T1 = {1./Sigma};
D = @(x) oper_grid(x, 1./Sigma, Mask, [Ny, Nx]);
Dt = @(x) oper_grid_adjoint(x, 1./Sigma, Mask, [Ny, Nx]);
evl1 = op_norm(D, Dt, [Ny, Nx], 1e-4, 200, verbosity);
% evl = max(1./Sigma)^2; 

T = mat2cell([1], 1);
W = mat2cell(true(size(yTmat)), length(yTmat));

param_pdfb.nu2 = evl1; % bound on the norm of the operator A*G
param_pdfb.gamma = 1e0; % convergence parameter L1 (soft th parameter)

result_st1 = [];
result_st1.sol = cell(num_tests, 1);
result_st1.L1_v = cell(num_tests, 1);
result_st1.L1_vp = cell(num_tests, 1);
result_st1.L2_v = cell(num_tests, 1);
result_st1.L2_vp = cell(num_tests, 1);
result_st1.time = cell(num_tests, 1);
result_st1.delta_v = cell(num_tests, 1);
result_st1.sol_v = cell(num_tests, 1);
result_st1.sol_reweight_v = cell(num_tests, 1);
result_st1.snr_v = cell(num_tests, 1);

result_st1.snr = cell(num_tests, 1);
result_st1.sparsity = cell(num_tests, 1);
result_st1.no_itr = cell(num_tests, 1);


for i = 1:num_tests
    % wavelet mode is a global variable which does not get transfered
    % to the workes; we need to set it manually for each worker
    dwtmode('per');

    fprintf('Test run %i:\n', i);

    tstart_a = tic;
    fprintf(' Running pdfb_bpcon_par_sim_rescaled\n');
 
    [result_st1.sol{i}, result_st1.L1_v{i}, result_st1.L1_vp{i}, result_st1.L2_v{i}, ...
        result_st1.L2_vp{i}, result_st1.delta_v{i}, result_st1.sol_v{i}, result_st1.snr_v{i}, ~, ~, result_st1.sol_reweight_v{i}] ...
        = pdfb_bpcon_par_sim_rescaled_adapt_eps(yT{i}, epsilonT{i}, epsilonTs{i}, epsilon{i}, epsilons{i}, D, Dt, T, W, Psi, Psit, Psiw, Psitw, param_pdfb);

    tend = toc(tstart_a);
    fprintf(' pdfb_bpcon_par_sim_rescaled runtime: %ds\n\n', ceil(tend));

    result_st1.time{i} = tend;
    
    error = im - result_st1.sol{i};
    result_st1.snr{i} = 20 * log10(norm(im(:))/norm(error(:)));
    
    result_st1.no_itr{i} = length(result_st1.L1_v{i});

    wcoef = [];
    for q = 1:length(Psit)
        wcoef = [wcoef; Psit{q}(result_st1.sol{i})];
    end
    result_st1.sparsity{i} = sum(abs(wcoef) > 1e-3)/length(wcoef);
end




function y = oper_grid(x, Sigma, Mask, imsize)
Fx =fftshift(fft2(ifftshift(reshape(x, imsize))));
Fx = Fx(:);
y = Sigma.*Fx(Mask);
end

function y = oper_grid_adjoint(x, Sigma, Mask, imsize)
Fy = zeros(prod(imsize),1);
Fy(Mask) = Sigma.* x;
y = fftshift(ifft2(ifftshift(reshape(Fy, imsize))));
end