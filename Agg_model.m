function  [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11] = Agg_model(inputdata, params)
    % X: mid-level deep map
    % Y: high-level deep map
    % img: RGB image
    % params: parameters for img
    % H1-H11: Deep structures histogram base on color map,
    % edge orientation map, intensity map, semantic map using DOS (Deep Orientation Space).
    
    X = inputdata{1};
    Y = inputdata{2};
    img = inputdata{end};
    
    [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11] = deal(0);
    
    [h1, w1, k1] = size(X);
    [h2, w2, k2] = size(Y);
    
    %% local general deep space, a map from mid-level features
    
    X(X<0) = 0;
    X_space = mean(X, 3);
    X_space = X_space ./ (sum(X_space.^2, 'all').^(1/2) + 1e-5);
    
    XX = X .* X_space;
    
    %% global perception by texton from low-level features

    X_map = globalPerceptionTexton(X_space, img, params);
    
    XZ = X .* X_map;
    
    %% local semantic perception by full convolution from high-level features
    
    Y(Y<0) = 0;

    Y_s = mean(Y.^6, 3).^(1/2);
%     Y_s = Y_s ./ (sum(Y_s.^2, 'all').^(1/2) + 1e-5);
    
%     Y_k = mean(Y.^2, [1 2]).^2;
%     Y_k = imresize3(Y_k, [1 1 k1]);
%     Y_k = reshape(Y_k,1,1,k2/k1,[]);
%     Y_k = mean(Y_k, 3);
%     Y_k = permute(Y_k, [1 2 4 3]);
    
    XYZ_D = convn(XZ, Y_s, 'same');
%     XYZ_map = convn(XZ, Y_k, 'valid');
%     XYZ_P = XYZ_D .* XYZ_map;
    
    XY_D = convn(X, Y_s, 'same');
%     XY_map = convn(X, Y_k, 'valid');
%     XY_P = XY_D .* XY_map;
    
    %% Representation of features, namely FVDH    
    
%     subplot(1, 6, 1); showHeatmap(img, X);
%     subplot(1, 6, 2); showHeatmap(img, Y);
%     subplot(1, 6, 3); showHeatmap(img, Y_s);
%     subplot(1, 6, 4); showHeatmap(img, XZ);
%     subplot(1, 6, 5); showHeatmap(img, XYZ_D);
%     subplot(1, 6, 6); showHeatmap(img, XY_D);
    
    H2 = sum(inputdata{1}, [1 2]);
    H3 = sum(inputdata{2}, [1 2]);
%    H5 = sum(Y, [1 2]);
    H6 = sum(XX, [1 2]);
    H7 = sum(XZ, [1 2]);
    H8 = sum(XYZ_D, [1 2]);
%     H9 = sum(XYZ_P, [1 2]);
    H10 = sum(XY_D, [1 2]);
%     H11 = sum(XY_P, [1 2]);
    
end
function Zmap = globalPerceptionTexton(X_space, img, params)
    % deep texton structures detection
    [cn1, cn2, cn3, CSO, CSI, CSS] = deal(params{1}, params{2}, params{3}, params{4}, params{5}, params{6});
    
    [h1, w1, ~] = size(X_space);
    [h2, w2, ~] = size(img);
    
    % RGB to HSV color space
    hsv = single(rgb2hsv(uint8(img)));
    
    % color map
    colormap = ColorQuantization(hsv, cn1, cn2, cn3);
    
    X_space = imresize(X_space, [h2, w2]);
    
    [T1, T2, T3, T4] = deal(single(zeros(h2, w2)));
    for h = 2:1:h2-1
        for w = 2:1:w2-1
            %color
            if colormap(h,w) == colormap(h,w+1)
                T1(h,w) = X_space(h,w);
                T1(h,w+1) = X_space(h,w+1);
            end
            if colormap(h,w) == colormap(h+1,w)
                T2(h,w) = X_space(h,w);
                T2(h+1,w) = X_space(h+1,w);
            end
            if colormap(h,w) == colormap(h+1,w+1)
                T3(h,w) = X_space(h,w);
                T3(h+1,w+1) = X_space(h+1,w+1);
            end
            if colormap(h+1,w) == colormap(h,w+1)
                T4(h+1,w) = X_space(h+1,w);
                T4(h,w+1) = X_space(h,w+1);
            end
            
        end
    end
    localcolormap = T1 + T2 + T3 + T4;

    Zmap = imresize(localcolormap, [h1, w1]);
end

function [] = showHeatmap(imdata, X)
    % imdata,  rgb image.
    % X, H*W*C tensor.
    
    imshow(uint8(imdata));
    X = imresize(X, size(imdata, [1,2]));
    hold on;
    imagesc(mean(X,3),'AlphaData',0.5);
    colormap jet;
    hold off;
end