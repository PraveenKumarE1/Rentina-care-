function result = main_pipeline(inputImage, options)
%MAIN_PIPELINE Run quality control, segmentation, grading, and evidence generation.
%   RESULT = MAIN_PIPELINE(FILE_OR_ARRAY) accepts a filename or RGB image.
arguments
    inputImage
    options.Language (1,1) string = "en"
    options.QualityThresholds struct = struct()
end
if ischar(inputImage) || isstring(inputImage)
    if ~isfile(inputImage), error('main_pipeline:MissingImage','Image not found: %s',inputImage); end
    image = imread(inputImage);
else
    image = inputImage;
end
[quality, enhanced] = image_quality(image, options.QualityThresholds);
segmentationResult = segmentation(enhanced, quality);
gradingResult = grading(segmentationResult, quality);
result = struct('quality',quality,'segmentation',segmentationResult,'grading',gradingResult, ...
    'image',enhanced,'summary',sprintf('%s: %s (confidence %.2f)', ...
    ternary(quality.passed,'Gradable','Ungradable'), gradingResult.label, gradingResult.confidence));
if quality.passed
        result.explainability = struct('lesionOverlay', lesion_overlay(enhanced,segmentationResult));
else
    result.explainability = struct('lesionOverlay', enhanced);
end
end
function value = ternary(condition, trueValue, falseValue)
if condition, value = trueValue; else, value = falseValue; end
end
