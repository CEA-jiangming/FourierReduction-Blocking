close all
clear variables
clc

usingReduction = 1;
normalize_data = 1;
usingPrecondition = 0;
enable_klargestpercent = 0;
klargestpercent = 50;
enable_estimatethreshold = 1;
gamma = 3;
ratio = 10;
input_snr = 40;

sigma_gaussian = [pi/16,pi/8];

addpath data
addpath data/images
addpath lib/
addpath fouRed/
% addpath src
addpath ../lib/irt/

try
    setup;
catch ME
    error('NUFFT library not found in location src/irt');
end

rng('shuffle');

for i = 1:length(sigma_gaussian)
    param_fouRed_uv_128{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma, ratio, input_snr, sigma_gaussian(i));
end

% result_precond_ratio50 = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma, ratio, input_snr);

% for i = 1:length(gamma)
%         param_fouRed_gamma_256{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma(i), ratio, input_snr, sigma_gaussian);
% end