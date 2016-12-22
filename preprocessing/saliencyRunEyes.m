% The following lines input a series of images into makeSaliencyMap( image, parameters) function
% implemented in 'SaliencyToolbox' by Walter. The processed files, i.e. saliency maps are computed and saved into a seperate folder. Matlab was a little tricky with paths, so I recommend providing absolute paths where paths are required.

filePattern = fullfile('eyes', '*.png'); % Change to whatever pattern you need.
imageFiles = dir(filePattern);
load('parameters.mat')
cnt = 0
for eye = 1: length(imageFiles)
    cnt=cnt+1;
    baseFileName = imageFiles(eye).name;
    fullFileName = fullfile('eyes', baseFileName);

    img.filename = fullFileName;
    Im = imread(img.filename);
    Im = imresize(Im,4);
    
    img.data = Im;
    img.dims = 3;
    img.type = 'unknown';
    img.size = size(Im);
    t = cputime;
    [salMap, salData]= makeSaliencyMap(img, params);
    S = strsplit(img.filename, '.');
    
    mapName = strcat(S(1), 'saliency.png');
    fullOut = fullfile('eyes/saliency', strjoin(mapName));
    imwrite(salMap.data, fullOut);
    if (mod(cnt, 100)==0)
        disp('+');
    end 
end

