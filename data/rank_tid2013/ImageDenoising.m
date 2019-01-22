function b= ImageDenoising(img, level)
     distorted_img = imnoise(img,'gaussian',0,level);
     
     yRGB = im2double(distorted_img); 
     % Generate the same seed used in the experimental results of [1]
     randn('seed', 0);
     % Standard deviation of the noise --- corresponding to intensity 
     %  range [0,255], despite that the input was scaled in [0,1]
     sigma = 25;
     % Add the AWGN with zero mean and standard deviation 'sigma'
     zRGB = yRGB + (sigma/255)*randn(size(yRGB));
     % Denoise 'zRGB'. The denoised image is 'yRGB_est', and 'NA = 1'  
     %  because the true image was not provided
     [~, yRGB_est] = CBM3D(1, zRGB, sigma); 
     % Compute the putput PSNR
%      PSNR = 10*log10(1/mean((yRGB(:)-yRGB_est(:)).^2))
     % show the noisy image 'zRGB' and the denoised 'yRGB_est'
%      figure; imshow(min(max(zRGB,0),1));   
%      figure; imshow(min(max(yRGB_est,0),1));
     b = uint8(yRGB_est*255);  

end