% Generate multiple distortions 
addpath('BM3D')  
file = dir('./pristine_images/*.bmp');
distortions = [1,2,5,6,7,8,9,10,11,14,15,16,17,18,19,22,23];
levels = 1:4;
for i = 1:length(file)

    refI = open_bitfield_bmp(fullfile('.', 'pristine_images', file(i).name));
    for type = distortions    % you could decide which types of distortion you want to generate
        for level = levels        % You could decide how many levels of distortion
            tid2013_generator(refI, type, level,file(i)); 
        end
    end
    fprintf('Finished image %d*16 / %d*16...\n', i,length(file));
    
end






