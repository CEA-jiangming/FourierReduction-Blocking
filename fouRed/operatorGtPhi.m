function y = operatorGtPhi(x, H, A, Sigma, Mask)
% New reduced measurement operator: Sigma * S * H * A
% Real -> Complex
tmp = H * A(x);
y = Sigma .* tmp(Mask);
