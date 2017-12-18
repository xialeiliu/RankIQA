function [out_im,status ] = open_bitfield_bmp( filename )
% OPEN_BITFIELD_BMP open a bitfield compressed bitmap image.
%
%   IM = OPEN_BITFIELD_BMP( FILENAME ) opens a 16-bit bitmapped compressed file 
%   named FILENAME. The output is formatted as three planes of Red, Green 
%   and Blue data. 
% 
%   See also imread, imfinfo

% Get information about the image
info = imfinfo(filename);

% if we have a bitfield compressed R-G-B bitmap image
if ( strcmp(info.Format , 'bmp') && ...
     strcmp(info.CompressionType , 'bitfields') &&  ...
     (info.BitDepth == 16 || info.BitDepth == 32) )

    % indicate the input bmp file is not of 8-bit
    status = 0;
 
    % Open the file for reading
    fid = fopen(filename, 'r');
        
    % Extract relevvant image info 
    data_offset = info.ImageDataOffset;
    width = info.Width;
    height = info.Height;
    
    % Create space for output image
    out_im = zeros(height, width, 3);
    
    % Seek to where the image data begins (i.e. skip the file header
    fseek(fid, data_offset, 'bof');
    
    % Read in the image data and format it into a matrix
    if (info.BitDepth == 16)
        % compressed_image = (fread(fid, [width + 1, height] , 'uint16'))';
        compressed_image = (fread(fid, [width, height] , 'uint16'))';
    else
        compressed_image = (fread(fid, [width, height] , 'uint32'))';
    end

    % Eliminate last column of junk data (scanline row terminators) in 16
    % bit mode
    if (info.BitDepth == 16)
        compressed_image = compressed_image(:, 1:(width));
    end;
        
    % Invert Row Order since it is flipped
    new_image = flipud(compressed_image);
    
    % Extract color bitmasks to decompress
    red_mask = info.RedMask;
    blue_mask = info.BlueMask;
    green_mask = info.GreenMask;
 
    
    % Extract color components and form output image
    out_im(:,:,1) = (bitand(new_image, red_mask) / red_mask * 255);
    out_im(:,:,2) = (bitand(new_image, green_mask) / green_mask * 255);
    out_im(:,:,3) = (bitand(new_image, blue_mask) / blue_mask * 255);

    % typecast
    out_im = uint8(out_im);

    % display image
    %imshow(out_im);
    
else

    % indicate the input bmp file is 8-bit
    status = 1;
    
    % general imread for all other cases
    out_im = imread(filename);
    %imshow(out_im);
    
end

