function [yPCAw, yPCA] = PCA_whitening(x_test, x_train, dim, method)
    %% method 1
    if nargin < 4
        method = 3;
    end
    if method == 1
        test = im_cross_normalize(x_test);
        test(isnan(test)) = 0;
        train = im_cross_normalize(x_train);        
        train(isnan(train)) = 0;

        x = train';
        avg = mean(x, 1);
        x = x - repmat(avg, size(x,1), 1);
        x(isnan(x)) = 0;
        sigma = (x*x') / size(x,2); 
        [U,~,~] = svd(sigma);

        y = test';
        avg = mean(y, 1);
        y = y - repmat(avg, size(y,1), 1);
        y(isnan(y)) = 0;
        Xpca = U(:, 1:dim)' * y; 
        sigma = (Xpca * Xpca') / size(Xpca,2);
        [u,s,~] = svd(sigma);
        xRot = u'* Xpca;
        epsilon = 1e-5;

        xPCAWhite = diag(1 ./ (sqrt(diag(s) + epsilon))) * xRot;
        x2 = xPCAWhite';

        yPCA = im_cross_normalize(xRot(1:dim,:)');
        yPCAw = im_cross_normalize(x2);
    end

    %% method 2
    
    if method == 2
        test = im_cross_normalize(x_test);
        test(isnan(test)) = 0;
        train = im_cross_normalize(x_train);
        train(isnan(train)) = 0;

        [coeff, scoreTrain, ~, ~, ~, mu] = pca(train); % PCA training
    %     scoreTrain = scoreTrain(:, 1:dim);   % train PCA, optional
    %     x_train = scoreTrain * coeff' + mu;      % the relation of the four
        sigma = scoreTrain' * scoreTrain / size(scoreTrain, 1);
        sigma(isnan(sigma)) = 0;
        [u, s, ~] = svd(sigma);

        scoreTest = (test - mu) * coeff;   % PCA apply
    %     scoreTest = scoreTest(:, 1:dim);   % test PCA, optional
        x_testRot = scoreTest * u;
        epsilon = 1e-5;
        x_testPCAWhite = x_testRot * diag(1 ./ (sqrt(diag(s) + epsilon)));    % whiten apply
        features_data = x_testPCAWhite(:, 1:dim);

        yPCA = im_cross_normalize(x_testRot(:,1:dim));
        yPCAw = im_cross_normalize(features_data);
    end
    
    %% method 3
    % from http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    if method == 3
        test = im_cross_normalize(x_test);
        test(isnan(test)) = 0;
        train = im_cross_normalize(x_train);        
        train(isnan(train)) = 0;

        x = train';
        avg = mean(x, 1);
        x = x - avg;
        x(isnan(x)) = 0;
        sigma = x * x' / size(x,2); 
        [U,S,~] = svd(sigma);       % U, S are the parameters of learning

        y = test';
        avg = mean(y, 1);
        y = y - avg;
        y(isnan(y)) = 0;
        yRot = U'* y;
        yPCA = yRot(1:dim, :)';

        epsilon = 1e-5;
        p = 2;
        yPCAWhite = diag(1 ./ ((diag(S) + epsilon).^(1/p))) * yRot;
        yPCAw = yPCAWhite(1:dim, :)';

        yPCA = im_cross_normalize(yPCA);
        yPCAw = im_cross_normalize(yPCAw);
    end

end