function eps_t = factor_eps(Uhat, Atrue)
% This function computes errors between estimated mode matrices and true
% ones
%
% Inputs:
% - Uhat: (p(k) x R) estimated mode-k matrix -> d cells
% - Atrue:  (p(k) x R) true mode-k matrix -> d cells
%
% Outputs:
% - eps_t: (scalar) error metric

d = numel(Atrue);
R = size(Atrue{1}, 2);

% Build aggregated R x R matching cost matrix
costMat = zeros(R, R);

for k = 1:d
    U = Uhat{k};
    A = Atrue{k};

    % Pairwise dot products: (i,j) = u_i' * a_j
    G = U' * A; % R x R
    uu = sum(U.^2, 1)'; % R x 1, ||u_i||^2
    aa = sum(A.^2, 1); % 1 x R, ||a_j||^2

    % Squared distances
    Dm2 = uu + aa - 2*G; % ||u_i - a_j||^2
    Dp2 = uu + aa + 2*G; % ||u_i + a_j||^2

    Dm = sqrt(max(Dm2, 0));
    Dp = sqrt(max(Dp2, 0));

    costMat = costMat + min(Dm, Dp);   % sum over modes
end

% Find permutation that minimizes total matching cost
if R <= 10
    % Brute force (fast enough for small R)
    P = perms(1:R);
    total = zeros(size(P,1), 1);
    for t = 1:size(P,1)
        total(t) = sum(costMat(sub2ind([R R], 1:R, P(t,:))));
    end
    [~, bestIdx] = min(total);
    perm = P(bestIdx,:);
else
    % Hungarian via matchpairs if available (Statistics and ML Toolbox)
    if exist('matchpairs', 'file') == 2
        pairs = matchpairs(costMat, Inf); % minimizes cost
        perm = zeros(1,R);
        perm(pairs(:,1)) = pairs(:,2);
    else
        error(['R is large and matchpairs() is not available. ', ...
               'Either install Statistics and Machine Learning Toolbox, ', ...
               'or keep R <= 10 to use brute force.']);
    end
end

% Compute epsilon_t after applying permutation
eps_t = 0;
for k = 1:d
    U = Uhat{k};
    A = Atrue{k}(:, perm);

    e1 = vecnorm(U - A, 2, 1);
    e2 = vecnorm(U + A, 2, 1);
    eps_t = max(eps_t, max(min(e1, e2)));
end

end