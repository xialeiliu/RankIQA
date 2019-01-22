function img2 = HighFN(img,level)

imgFFT = fft2(double(img));
thresh = 10;
b(:,:,1) = ghp(imgFFT(:,:,1),thresh);
b(:,:,2) = ghp(imgFFT(:,:,2),thresh);
b(:,:,3) = ghp(imgFFT(:,:,3),thresh);

img2 = ifft2(b);
img2 = real(img2);
img2= uint8(img2);
img2 = imnoise(img2,'gaussian',0,level);

end