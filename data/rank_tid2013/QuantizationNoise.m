function b = QuantizationNoise(img, level)

threshRGB = multithresh(img,level);

threshForPlanes = zeros(3,level);			

for i = 1:3
    threshForPlanes(i,:) = multithresh(img(:,:,i),level);
end

value = [0 threshRGB(2:end) 255]; 
b = imquantize(img, threshRGB, value);

end