function b= MultiGN(img_rgb,level)

img_rgb = im2double(img_rgb)*255;
noiseOnlyImage = 1+level* randn(size(img_rgb,1),size(img_rgb,2));
img_rgb(:,:,1) = img_rgb(:,:,1) .* noiseOnlyImage;

noiseOnlyImage = 1+level* randn(size(img_rgb,1),size(img_rgb,2));
img_rgb(:,:,2) = img_rgb(:,:,2) .* noiseOnlyImage;

noiseOnlyImage = 1+level * randn(size(img_rgb,1),size(img_rgb,2));
img_rgb(:,:,3) = img_rgb(:,:,3) .* noiseOnlyImage;

b=uint8(img_rgb);

end