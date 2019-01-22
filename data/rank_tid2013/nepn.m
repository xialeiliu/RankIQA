function b = nepn(img, level)

W = size(img,1);
H = size(img,2);

img_out = img;

for i=1:level
    r_W = randi([1 W-30],1);
    r_H = randi([1 H-30],1);
    img_out(r_W:r_W+14,r_H:r_H+14,:) = img(r_W+5:r_W+19,r_H+5:r_H+19,:);

end
b= img_out;

end