function [Z, IDF] = IDF(X)
    % IDF: inverse document frequency.
    % X: tensor, 1*D, or 1*1*D or 1*1*1*D...
    % IDF: tensor, 1*D, or 1*1*D or 1*1*1*D...
    % Z: tensor, 1*D, or 1*1*D or 1*1*1*D...
    
    epsilon = 1e-5;
    n = ndims(X);
    DF = abs(X);
    IDF = log((sum(DF, n) + epsilon) ./ (DF + epsilon));
    Z = X .* IDF;
end