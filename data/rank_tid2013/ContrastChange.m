function b = ContrastChange(img,level)

img = im2double(img)*255;
img= img*level;
b = uint8(img);
end