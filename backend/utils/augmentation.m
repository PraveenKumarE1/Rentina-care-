function output=augmentation(inputImage,seed)
%AUGMENTATION Apply deterministic, mild training-time augmentation.
if nargin>1,rng(seed);end
output=inputImage;
if rand>0.5,output=fliplr(output);end
if rand>0.5,output=imadjust(output,[],[],0.9+0.2*rand);end
end
