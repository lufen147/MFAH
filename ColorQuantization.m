function colormap = ColorQuantization(HSV, colnum1,  colnum2, colnum3)
    
    HI = fix(HSV(:,:,1) .* colnum1);
    HI(HI>(colnum1-1)) = colnum1 - 1;
    HI(HI<0) = 0;
    
    SI = fix(HSV(:,:,2) .* colnum2);
    SI(SI>(colnum2-1)) = colnum2 - 1;
    SI(SI<0) = 0;

    VI = fix(HSV(:,:,3) .* colnum3);
    VI(VI>(colnum3-1)) = colnum3 - 1;
    VI(VI<0) = 0;
    
    colormap = (colnum3 .* colnum2) .* HI + colnum3 .* SI + VI + 1; 
end