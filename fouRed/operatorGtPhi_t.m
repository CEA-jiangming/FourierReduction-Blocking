function y = operatorGtPhi_t(x, H, At, Sigma, Mask, Kd)
% Adjoint of the new reduced measurement operator: At * H * S * Sigma
% Complex -> Real
Ny = Kd(1);
Nx = Kd(2);
x1 = zeros(Ny * Nx, 1);
x1(Mask) = Sigma .* x(:);
y = real(At(H * x1));
