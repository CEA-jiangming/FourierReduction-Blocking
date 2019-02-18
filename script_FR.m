close all
clear variables
clc

usingReduction = 0;
normalize_data = 0;
usingPrecondition = 1;
enable_klargestpercent = 1;
klargestpercent = 25;
enable_estimatethreshold = 0;
gamma = 1:5;
ratio = [25,50,100];

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

for i = 1:length(ratio)
    result_0FR_1precond{i} = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma, ratio(i));
end

% result = parallel_FR_pd(usingReduction, normalize_data, usingPrecondition, enable_klargestpercent, klargestpercent, enable_estimatethreshold, gamma)