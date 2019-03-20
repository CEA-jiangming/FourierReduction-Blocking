close all
clear variables
clc

usingReduction = 1;
normalize_data = 1;
usingPrecondition = 0;
enable_klargestpercent = 1;
klargestpercent = 50;
enable_estimatethreshold = 0;
gamma = [1,10:10:100];
ratio = 5;
input_snr = 20;

sigma_gaussian = pi/4;

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

% for i = 1:length(klargestpercent)
%     result_gamma_128{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent(i), enable_estimatethreshold, gamma, ratio, input_snr, sigma_gaussian);
% end

result_precond_ratio50 = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma, ratio, input_snr);

% for i = 1:length(gamma)
%         param_fouRed_gamma_256{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma(i), ratio, input_snr, sigma_gaussian);
% end