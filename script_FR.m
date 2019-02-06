close all
clear variables
clc

usingReduction = 1;
normalize_data = 1;
usingPrecondition = 1;
enable_klargestpercent = 0;
klargestpercent = [1, 5, 10, 25, 50, 100];
enable_estimatethreshold = 1;
gamma = 200;

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

for i = 1:length(klargestpercent)
    result_klargestpercent_precond{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent(i), enable_estimatethreshold, gamma)
end

% result = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma)