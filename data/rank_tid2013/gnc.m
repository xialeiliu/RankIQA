function b = gnc(img_rgb,level)

ycbcr = rgb2ycbcr(img_rgb);
ycbcr = im2double(ycbcr);
sizeA = size(ycbcr);
b = ycbcr + sqrt(level)*randn(sizeA);
b=ycbcr2rgb(b)*255;
b=uint8(b);

end