function [quality, enhanced] = image_quality(inputImage, thresholds)
%IMAGE_QUALITY Assess focus, illumination, field of view, noise, and contrast.
if nargin < 2, thresholds = struct(); end
thresholds = defaults(thresholds, struct('minFocus',25,'minContrast',0.08,'minFovDiameter',0.45,'minSnr',3));
if size(inputImage,3) == 3, gray = im2double(rgb2gray(inputImage)); else, gray = im2double(inputImage); end
gray = mat2gray(gray);
focus = var(imfilter(gray,[-1 -1 -1; -1 8 -1; -1 -1 -1],'replicate'),0,'all');
contrast = std(gray(:));
margin = max(1,round(min(size(gray))*0.05));
border = [gray(1:margin,:) ; gray(end-margin+1:end,:) ; gray(:,1:margin)'; gray(:,end-margin+1:end)'];
center = gray(round(end/4):round(3*end/4),round(size(gray,2)/4):round(3*size(gray,2)/4));
vignetting = mean(center(:))/(mean(border(:))+eps);
fovMask = gray > 0.05;
fovDiameter = max(size(gray))*sqrt(nnz(fovMask)/numel(fovMask));
fovDiameter = fovDiameter/max(size(gray));
snr = mean(gray(:))/(std(gray(:))+eps);
quality = struct('focus',focus,'contrast',contrast,'vignettingRatio',vignetting, ...
    'fieldOfView',fovDiameter,'snr',snr,'passed',focus >= thresholds.minFocus && ...
    contrast >= thresholds.minContrast && fovDiameter >= thresholds.minFovDiameter && snr >= thresholds.minSnr);
quality.feedback = strings(0,1);
if focus < thresholds.minFocus, quality.feedback(end+1) = "Poor focus - retake"; end
if contrast < thresholds.minContrast, quality.feedback(end+1) = "Low contrast - retake"; end
if fovDiameter < thresholds.minFovDiameter, quality.feedback(end+1) = "Insufficient field of view - retake"; end
if vignetting < 0.65 || vignetting > 1.8, quality.feedback(end+1) = "Uneven illumination"; end
if isempty(quality.feedback), quality.feedback = "Pass"; end
enhanced = inputImage;
if ~quality.passed && contrast >= thresholds.minContrast*0.7
    if size(inputImage,3)==3
        enhanced = cat(3, adapthisteq(inputImage(:,:,1)),adapthisteq(inputImage(:,:,2)),adapthisteq(inputImage(:,:,3)));
    else, enhanced = adapthisteq(inputImage); end
end
end
function output = defaults(input, template)
output = template; names = fieldnames(input); for k=1:numel(names), output.(names{k})=input.(names{k}); end
end
