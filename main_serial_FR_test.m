close all
clear variables
clc

visibSize = 5 * 256 * 256;
input_snr = 20;
image_file_name = './data/images/M31_256.fits';
coveragefile = '.data/vis/uv.fits';
klargestpercent = 50;  % Percent of image size to keep after dimensionality reduction
run = 1;
normalize_data = 1;

result_snr = [];
result_time = [];
epsilon_global = [];

addpath data
addpath data/images
addpath lib/
addpath fouRed/
addpath irt/

try
    setup;
catch ME
    error('NUFFT library not found in location src/irt');
end

rng('shuffle');

%% run parameters
% 0 - loads new data from file based on the dataset number supplied
% 1 - generates new data
% 2 - uses the data in matlab's workspace
gen_data = 1;
gen_figures = 1;
gen_only_average_figures = 0;
free_memory = 0;

save_dataset_number = 5; % number of the dataset to write files to
save_dataset_subnumber = 0; % number of the dataset to write files to

save_data_on_disk = 0; % flag
save_eps_files = 0; % flag
save_path = 'results/rsing/';

num_tests = 10;

%% various config parameters
verbosity = 1;

ox = 2; % oversampling factors for nufft
oy = 2; % oversampling factors for nufft
Kx = 8; % number of neighbours for nufft
Ky = 8; % number of neighbours for nufft

use_gridded_data = 0; % flag setting for generating gridded data

% evl params

% compute_evl = 0;
% compute_evl_no_natw = 0;
% compute_evl_precond = 0;
% compute_block_op_norm = 0; % flag to compute the operator norm for each block
% 
% use_symmetric_fourier_sampling = 0;


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
                             
%% sparsity prior
wlt_basis = {'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'self'}; % wavelet basis to be used
nlevel = 4; % wavelet level

%% nufft parameters

param_nufft.gen_fft_op_without_scale = 0;
param_nufft.use_fft_mask = 1;
param_nufft.use_fft_on_gpu = 0; % gpu FFT
param_nufft.use_nufft_blocks = 0;

%% Fourier reduction parameters
param_fouRed.klargestpercent = klargestpercent;  
param_fouRed.diagthresholdepsilon = 1e-10; 
param_fouRed.covmatfileexists = 0;
param_fouRed.covmatfile = 'covariancemat.mat';
param_fouRed.fastCov = 1;

%% block structure

regenerate_block_structure = 1;
param_block_structure.use_density_partitioning = 0;
param_block_structure.density_partitioning_no = 1;
param_block_structure.use_uniform_partitioning = 0;
param_block_structure.uniform_partitioning_no = 4;
param_block_structure.use_manual_frequency_partitioning = 0;
param_block_structure.fpartition = [icdf('norm', 0.25, 0, pi/4), 0, icdf('norm', 0.75, 0, pi/4), pi]; % partition (symetrically) of the data to nodes (frequency ranges)
param_block_structure.use_manual_partitioning = 0;
param_block_structure.partition = [1000 2000 4000];

param_block_structure.use_equal_partitioning = 1;
param_block_structure.equal_partitioning_no = 1;

%% Singular values based block structure
% param_sing_block_structure.use_uniform_partitioning = 0;
% param_sing_block_structure.use_sort_uniform_partitioning = 1;
% param_sing_block_structure.uniform_partitioning_no = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Generating new data ... \n\n');
%% image and uv 
[im, N, Ny, Nx] = util_read_image(image_file_name);

uvfile = './data/uv.mat';

if gen_data
    sampling_pattern = 'gaussian+large-holes';
    
    [im, N, Ny, Nx] = util_read_image(image_file_name);

    param_sampling.N = N; % number of pixels in the image
    param_sampling.Nox = ox*Nx; % number of pixels in the image
    param_sampling.Noy = oy*Ny; % number of pixels in the image
    util_gen_sampling_pattern_config; % Set all parameters

    [~, ~, uw, vw, ~] = util_gen_sampling_pattern(sampling_pattern, param_sampling);

    save(uvfile,'uw','vw')
else
    load(uvfile)
end

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

[Psi, Psit] = op_p_sp_wlt_basis(wlt_basis, nlevel, Ny, Nx);
[Psiw, Psitw] = op_sp_wlt_basis(wlt_basis, nlevel, Ny, Nx);

%% generate noisy input data
 
for k = 1:num_tests
    % cell structure to adapt to the previous solvers
    if normalize_data
        [y0{k}{1}, ~, y{k}{1}, ~, sigma_noise,~, noise{k}{1}] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    else
        [y0{k}{1}, y{k}{1}, ~, ~, sigma_noise, noise{k}{1}, ~] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    end        
end


%% For dimensionality reduction
    
% psf operator Ipsf, singular value matrix Sigma, mask matrix (to reduce the dimension)
[Ipsf, Sigma, Mask] = fourierReduction(Gw, A, At, [Ny, Nx], param_fouRed);
% New measurement operator C, new reduced measurement operator B
[C, Ct, B, Bt] = oper_fourierReduction(Ipsf, Sigma, Mask, [Ny, Nx]);

evl = op_norm(B, Bt, [Ny, Nx], 1e-4, 200, verbosity);

% Embed the y using the same reduction
for k = 1:num_tests
    ry = fftshift(fft2(ifftshift(At(Gw'*y{k}{1}))));
    ry = ry(:);
    yTmat = Sigma.*ry(Mask);
  
    % This section is to adapt to the current code structure 
    T = mat2cell([1], 1);
    W = mat2cell(true(size(yTmat)), length(yTmat));
    yT{k} = mat2cell(yTmat, length(yTmat));
    
end

%Bound for the L2 norm
fprintf('Computing epsilon bound... ');
tstart1=tic;      

% Embed the noise
for k = 1:num_tests
    % Apply F Phi
    rn = fftshift(fft2(ifftshift(At(Gw'*noise{k}{1}))));
    rn = rn(:);

    % factorized by singular values
    epsilon{k} = norm(Sigma .* rn(Mask));
    % epsilon = step_epsilon; % set epsilon value BEFORE running this script
    epsilons{k} = 1.001*epsilon{k};     % data fidelity error * 1.001
%         histogrampeakiness = mean(d12rnnorms)/std(d12rnnorms);
    epsilonT{k}{1} = epsilon{k};
    epsilonTs{k}{1} = epsilons{k};

end
    %%%%%%%%%%%%%%%
fprintf('Done\n');
tend1=toc(tstart1);
fprintf('Time: %e\n', tend1);

%% PDFB parameter structure sent to the algorithm
param_pdfb.im = im; % original image, used to compute the SNR
param_pdfb.verbose = verbosity; % print log or not
param_pdfb.nu1 = 1; % bound on the norm of the operator Psi
param_pdfb.nu2 = evl; % bound on the norm of the operator A*G
param_pdfb.gamma = 1e-3; % convergence parameter L1 (soft th parameter)
param_pdfb.tau = 0.49; % forward descent step size
param_pdfb.rel_obj = 1e-5; % stopping criterion
param_pdfb.max_iter = 500; % max number of iterations
param_pdfb.lambda0 = 1; % relaxation step for primal update
param_pdfb.lambda1 = 1; % relaxation step for L1 dual update
param_pdfb.lambda2 = 1; % relaxation step for L2 dual update
param_pdfb.sol_steps = inf; % saves images at the given iterations

param_pdfb.use_proj_elipse_fb = 1;
param_pdfb.elipse_proj_max_iter = 10;
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

param_pdfb.use_adapt_bound_eps = 0;
param_pdfb.adapt_bound_steps = 100;
param_pdfb.adapt_bound_rel_obj = 1e-5;
param_pdfb.hard_thres = 0;
param_pdfb.adapt_bound_tol =1e-3;
param_pdfb.adapt_bound_start = 1000;

param_pdfb.savepath = save_path;


%% compute the solution
fprintf('Starting algorithms:\n\n');

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


for i = 1:num_tests
    % wavelet mode is a global variable which does not get transfered
    % to the workes; we need to set it manually for each worker
    dwtmode('per');

    fprintf('Test run %i:\n', i);

    tstart_a = tic;
    fprintf(' Running pdfb_bpcon_par_sim_rescaled\n');
        
    [result_st.sol{i}, result_st.L1_v{i}, result_st.L1_vp{i}, result_st.L2_v{i}, ...
            result_st.L2_vp{i}, result_st.delta_v{i}, result_st.sol_v{i}, result_st.snr_v{i}, ~, ~, result_st.sol_reweight_v{i}] ...
            = pdfb_bpcon_par_sing_sim_rescaled_adapt_eps(yT{i}, [Ny, Nx], epsilonT{i}, epsilonTs{i}, epsilon{i}, epsilons{i}, C, Ct, T, W, Psi, Psit, Psiw, Psitw, param_pdfb);
        
    if normalize_data
       result_st.sol{i} = result_st.sol{i}*sigma_noise/sqrt(2);
    end

    tend = toc(tstart_a);
    fprintf(' pdfb_bpcon_par_sing_sim_rescaled runtime: %ds\n\n', ceil(tend));

    result_time(i) = tend;
    error = im - result_st.sol{i};
    result_snr(i) = 20 * log10(norm(im(:))/norm(error(:)));
end
