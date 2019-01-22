function b= CA(img_rgb,level)
 
 hsize=3;

 R=(img_rgb(:,:,1));
 G=(img_rgb(:,:,3));
 B=(img_rgb(:,:,2));
 R2=R;
 B2=B;
 R2(:,level:end)=R(:,1:end-level+1);
 B2(:,level/2:end)=B(:,1:end-level/2+1);
 temp = img_rgb;
 temp(:,:,1)=R2;
 temp(:,:,2)=B2;
 img=temp;
 h = fspecial('gaussian', hsize, hsize/6);
 b=imfilter(img,h,'symmetric');
 

end