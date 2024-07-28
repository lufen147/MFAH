function Z = im_cross_normalize(X)
    % im_cross_normalize: A helper function that wraps the function of the same name in sklearn.
    %     This helper handles the case of a single column vector.
    % iuput:
    %   X: any type data
    % output
    %   Z: normalize's data
    
    % Z = normalize(X, 2, 'norm', 2);  % normalize x each row(the first 2 represent) with 2-norm
    Z = normalize(X, 2, 'norm');
end