function  distortion_generator( img, dist_type, level, filename )
    %% set distortion parameter
    gblur_level = [7,15,39,91,199];
    wn_level = [-10,-7.5,-5.5,-3.5,0];
    jpeg_level = [43,12,7,4,0];
    jp2k_level = [0.46,0.16,0.07,0.04,0.02]; % bit per pixel
    
    %% distortion generation
    switch dist_type
        case 1
            strname = './GB/GB';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            hsize = gblur_level(level);
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img = imfilter(img,h,'symmetric');
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 2
            strname = './GN/GN';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            distorted_img = imnoise(img,'gaussian',0,2^(wn_level(level)));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 3
            strname = './JPEG/JPEG';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.jpg']);
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(img,testName,'jpg','quality',jpeg_level(level));
        case 4
            strname = './JP2K/JP2K';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.jp2']);
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(img,testName,'jp2','CompressionRatio', 24 / jp2k_level(level));
    end
end

