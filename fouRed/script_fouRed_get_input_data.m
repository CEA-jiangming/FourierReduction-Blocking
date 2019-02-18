%% generate the sampling pattern
    
if gen_data == 0
    fprintf('Loading data from disk ... \n\n');
    load(uvfile)
    
elseif gen_data == 1
    [im, N, Ny, Nx] = util_read_image(image_file_name);
    global im;

    param_sampling.N = N; % number of pixels in the image
    param_sampling.Nox = ox*Nx; % number of pixels in the image
    param_sampling.Noy = oy*Ny; % number of pixels in the image
    util_gen_sampling_pattern_config; % Set all parameters

    [~, ~, uw, vw, ~] = util_gen_sampling_pattern(sampling_pattern, param_sampling);


    if use_symmetric_fourier_sampling
        uw = [uw; -uw];
        vw = [vw; -vw];
    end
    
    if save_data_on_disk == 1
        fprintf('Saving new data ... \n');
        
        if save_data_on_disk
            save(uvfile,'uw','vw')
        end
    end
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
    % cell structure to adapt to the solvers
    if normalize_data
%         G = spdiags(sqrt(2)/sigma_noise, 0, b_l, b_l) * G;
%         Gw = spdiags(sqrt(2)/sigma_noise, 0, b_l, b_l) * Gw;
        [y0{k}{1}, ~, y{k}{1}, ~, sigma_noise,~, noise{k}{1}] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    else
        [y0{k}{1}, y{k}{1}, ~, ~, sigma_noise, noise{k}{1}, ~] = util_gen_input_data_noblock(im, G, W, A, input_snr);
    end        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For dimensionality reduction

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
    ry = fftshift(fft2(ifftshift(At(Gw'*y{k}{1}))));
    ry = ry(:);
    yTmat = Sigma.*ry(Mask);

    clear T W;
    if usingReductionPar
        [yT{k}, T, W] = util_gen_sing_block_structure(yTmat, Sigma, Mask, param_sing_block_structure);
        if usingPrecondition
            R = length(W);
            aW = cell(R,1);
            for q = 1:R
                aW{q} = 1./T{q};
            end
        end
    else
        % This section is to adapt to the current code structure 
        T = {Sigma};
        W = {Mask};
        yT{k} = {yTmat};
        if usingPrecondition
            aW = {1./T{1}};
        end
    end
end

if usingPrecondition
    evl = op_norm(@(x) sqrt(cell2mat(aW)) .* B(x), @(x) Bt(sqrt(cell2mat(aW)) .* x), [Ny, Nx], 1e-6, 200, verbosity);
else
    evl = op_norm(B, Bt, [Ny, Nx], 1e-4, 200, verbosity); 
end

%Bound for the L2 norm
fprintf('Computing epsilon bound... ');
tstart1=tic;      

% Embed the noise
for k = 1:num_tests
    % Apply F Phi
    rn = fftshift(fft2(ifftshift(At(Gw'*noise{k}{1}))));
    rn = rn(:);
    
    % factorized by singular values and compute l2 ball       
    for i = 1:length(T)
        epsilonT{k}{i} = norm(T{i} .* rn(W{i}));
        epsilonTs{k}{i} = 1.001*epsilonT{1}{i};
    end
    epsilon{k} = norm(cell2mat(epsilonT{k}));
    epsilons{k} = 1.001*epsilon{k};     % data fidelity error * 1.001
end

    %%%%%%%%%%%%%%%
fprintf('Done\n');
tend1=toc(tstart1);
fprintf('Time: %e\n', tend1);
