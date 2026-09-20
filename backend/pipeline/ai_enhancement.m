function enhanced = ai_enhancement(inputImage)
%AI_ENHANCEMENT Apply deterministic, clinically conservative image enhancement.
%   The pipeline corrects illumination and contrast while preserving lesion
%   morphology. It is used by both MATLAB and Python screening paths.

if size(inputImage, 3) == 1
    gray = im2single(inputImage);
    gray = mat2gray(gray);
    illumination = imgaussfilt(gray, max(10, round(min(size(gray)) / 16)));
    corrected = mat2gray(gray ./ (illumination + 0.05));
    corrected = adapthisteq(corrected, 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
    enhanced = im2uint8(corrected);
    return
end

rgb = im2single(inputImage);
rgb = mat2gray(rgb, [], [], 'omitnan');
lab = rgb2lab(rgb);
lab(:, :, 1) = adapthisteq(lab(:, :, 1), 'ClipLimit', 0.02, 'Distribution', 'rayleigh');
rgb = lab2rgb(lab);

gray = rgb2gray(rgb);
illumination = imgaussfilt(gray, max(10, round(min(size(gray)) / 16)));
correctedGray = mat2gray(gray ./ (illumination + 0.05));
ratio = correctedGray ./ (gray + 0.02);
ratio = mat2gray(ratio);
for channel = 1:3
    rgb(:, :, channel) = mat2gray(rgb(:, :, channel) .* ratio);
end

for channel = 1:3
    rgb(:, :, channel) = medfilt2(rgb(:, :, channel), [3 3]);
end
sharp = imsharpen(rgb, 'Amount', 0.35, 'Radius', 1);
enhanced = im2uint8(min(max(sharp, 0), 1));
end
