function  tid2013_generator( img, dist_type, level, filename )
    %% set distortion parameter
    wn_level = [0.001, 0.005,0.01, 0.05]; % #1 Gaussian noise
    gnc_level = [0.0140,0.0198,0.0343,0.0524];     % #2 Gaussian noise in color components
    hfn_level = [0.001,0.005,0.01,0.05];      % #5 High frequency noise
    in_level = [0.005,0.01,0.05,0.1];     % #6 Impulse noise  
    qn_level = int32([255./27,255./39,255./55,255./76]);  % #7 Quantization noise
    gblur_level = [7,15,39,91];  % #8 Gaussian Blur
    id_level = [0.001, 0.005,0.01, 0.05];              % #9 Image Denoising          
    jpeg_level = [43,12,7,4];  % #10 JPEG compression
    jp2k_level = [0.46,0.16,0.07,0.04]; % #11  JP2K compression
    nepn_level = [30,70,150,300];  % #14  Non eccentricity pattern noise  
    bw_level = [2,4,8,16,32];    % #15  Local block-wise distortions of different intensity
    ms_level = [15,30,45,60] ;  % #16  Mean shift MSH =[15,30,45,60] MSL = [-15,-30,-45,-60]
    cc_level = [0.85,0.7,0.55,0.4];  % #17  Contrast change [0.85,0.7,0.55,0.4] [1.2,1.4,1.6,1.8]
    cs_level = [0.4,0,-0.4,-0.8];  % #18  color saturation
    mgn_level = [0.05, 0.09,0.13, 0.2];   % #19  Multiplicative Gaussian noise
    cqd_level = [64,32, 16,8,4];   % #22  Color quantization dither
    ca_level = [2,6,10,14];   % #23  Color aberrations
    %% distortion generation
    switch dist_type

        case 1
            strname = './GN/GN';

            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            distorted_img = imnoise(img,'gaussian',0,(wn_level(level)));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 2
            strname = './GNC/GNC';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = gnc(img,gnc_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 5
            strname = './HFN/HFN';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = HighFN(img,(hfn_level(level)));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 6
            strname = './IN/IN';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img =imnoise(img,'salt & pepper',(in_level(level)));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 7
            strname = './QN/QN';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = QuantizationNoise(img,qn_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 8
            strname = './GB/GB';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            hsize = gblur_level(level);
            h = fspecial('gaussian', hsize, hsize/6);
            distorted_img = imfilter(img,h,'symmetric');
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 9
            strname = './ID/ID';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = ImageDenoising(img,id_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 10
            strname = './JPEG/JPEG';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.jpg']);
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(img,testName,'jpg','quality',jpeg_level(level));
        case 11
            strname = './JP2K/JP2K';
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.jp2']);
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(img,testName,'jp2','CompressionRatio', 24 / jp2k_level(level));

        case 14
            strname = './NEPN/NEPN';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = nepn(img,nepn_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 15
            strname = './BW/BW';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = BlockWise(img,bw_level(level),level);
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
            
        case 16
            strname = './MSH/MSH';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = MeanShift(img,ms_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 17
            strname = './CCL/CCL';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = ContrastChange(img,cc_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 18
            strname = './CS/CS';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = ColorSaturation(img,cs_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 19
            strname = './MGN/MGN';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = MultiGN(img,mgn_level(level));
            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);
        case 22
            strname = './CQD/CQD';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            [temp,map]=rgb2ind(img,cqd_level(level));
            distorted_img = uint8(ind2rgb(temp,map)*255);

            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);

        case 23
            strname = './CA/CA';
            
            testName = fullfile([strname, int2str(level)],[filename.name(1:end-4),'.bmp']);
            
            distorted_img = CA(img,ca_level(level));

            if ~exist([strname, int2str(level)], 'dir')
                mkdir([strname, int2str(level)]);
            end
            imwrite(distorted_img,testName);


    end
end

