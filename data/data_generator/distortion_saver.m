% Generate different distortions 
file = dir('./pristine_images/*.bmp');   % The folder path of dataset

for i = 1:length(file)
    refI = open_bitfield_bmp(fullfile('.', 'pristine_images', file(i).name));
    for type = 1:4
        for level = 1:5
            distortion_generator(refI, type, level,file(i)); % #ok
        end
    end
    fprintf('Finished image %d*21 / 4744*21...\n', i);
end
