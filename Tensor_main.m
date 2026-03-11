%% MAIN
% This code checks convergence of CP-ALS decomposition via simulations
clc;
clear;

ttroot = 'C:\Users\violi\OneDrive\Desktop\PhD courses\Second year\TENSOR\tensor_toolbox-v3.8';
addpath(ttroot);

% Paramaters
rngSeed = 42;
R = 3; % CP-rank
d = 3;
p = [15 12 10]; % d x 1
lambda = [1 2 3]; % R x 1

M = 50; % Number of simulations
maxit = 100; % Maximum number of iterations
SNRdB = linspace(10,40, 10);
sigma = [10^(-9), 10^(-7), 10^(-5), 10^(-4), 10^(-2), 1];
xi = linspace(0, 0.9, 10);
N = size(xi,2);
S = size(sigma, 2);

% Initialize vectors
iters = zeros(N, S);
iters_MC = zeros(M,1);

% Generate random vectors for initialization
initSeed = 43;
rng(initSeed);
U0 = cell(d,1);
for n = 1:d
    U0{n} = rand(p(n), R);
end

% For parallel computing
% delete(gcp('nocreate'));
% parpool('local', 8);

for s = 1:S

    for i=1:N
    
        parfor j = 1:M
            seed = rngSeed + j; % We generate different tensors for each simulation

            [X, Y, A] = cp_tensor(R, d, p, xi(i), lambda, SNRdB(s), seed, sigma(s)); % Generate tensor

            eps_target = 0.05; % Target for error metric

            Ucur = U0; % Initial matrices

            t_final = maxit;

            for t = 1:maxit
                [P, ~, ~] = cp_als(Y, R, 'init', Ucur, 'maxiters', 1, 'tol', 0, 'printitn', 0); % Just one iteration of ALS

                Ucur = P.U; % Store the output
                eps_t = factor_eps(Ucur, A); % Compute error metric

                if eps_t <= eps_target % Check convergence
                    t_final = t;
                    break;
                end
            end

            iters_MC(j) = t_final;
        end
    
        % proportion of iters that converged
        iters(i,s) = sum(iters_MC < maxit)/M;

    end

end

%% Figures with sigma
% Against xi
figure;
plot(xi, iters(:,1), 'LineWidth', 1.5); hold on % sigma = 10^(-9)
plot(xi, iters(:, 4), 'LineWidth', 1.5); hold on % sigma = 10^(-4)
plot(xi, iters(:, 6), 'LineWidth', 1.5); % sigma = 1
xlabel('\xi');
ylabel('$\hat{\pi}$','Interpreter','latex');
ylim([0 1.0])
legend('\sigma = 10^{(-9)}', '\sigma = 10^{(-4)}', '\sigma = 1');
title('Iterations vs \xi for different \sigma values. R = 3 d = 3');
grid on;

exportgraphics(gcf, fullfile(pwd, 'Immagini_3', 'xi_R3_d3.jpg'));

% Against sigma
figure;
plot(sigma, iters(2, :), 'LineWidth', 1.5); hold on % xi = 0.10
plot(sigma, iters(8, :), 'LineWidth', 1.5); hold on % xi = 0.70
plot(sigma, iters(10, :), 'LineWidth', 1.5); % xi = 0.90
xlabel('\sigma');
ylabel('$\hat{\pi}$','Interpreter','latex');
ylim([0 1.0])
legend('\xi = 0.10', '\xi = 0.70', '\xi = 0.90');
title('Iterations vs \sigma for different \xi values. R = 3 d = 3');
grid on;

exportgraphics(gcf, fullfile(pwd, 'Immagini_3', 'sigma_R3_d3.jpg'));



