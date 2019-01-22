function b=BlockWise(img,level,flag)

R_mean = mean(mean(img(:,:,1)));
G_mean = mean(mean(img(:,:,2)));
B_mean = mean(mean(img(:,:,3)));
W = size(img,1);
H = size(img,2);

switch flag
    case 5
        block = repmat([R_mean G_mean B_mean]',[1,32,32]);
        block = permute(block,[3,2,1]);
    case 4
        block = repmat([R_mean G_mean B_mean]',[1,32,32]);
        block = permute(block,[3,2,1]);
    case 3
        block = repmat([R_mean+30 G_mean B_mean]',[1,32,32]);
        block = permute(block,[3,2,1]);
    case 2
        block = repmat([R_mean+50 G_mean B_mean]',[1,32,32]);
        block = permute(block,[3,2,1]);
    case 1
        block = repmat([0 G_mean B_mean]',[1,32,32]);
        block = permute(block,[3,2,1]);
end

block = uint8(block);

for i=1:level
    r_W = randi([1 W-32],1);
    r_H = randi([2 H-32],1);
    img(r_W:r_W+31,r_H:r_H+31,:) = block;
end

b= img;

end