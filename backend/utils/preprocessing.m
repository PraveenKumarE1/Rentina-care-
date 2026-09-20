function output=preprocessing(inputImage,targetSize)
%PREPROCESSING Normalize and resize a fundus image for model or classical analysis.
if nargin<2,targetSize=[224 224];end
if size(inputImage,3)==1,inputImage=repmat(inputImage,1,1,3);end
output=im2single(imresize(inputImage,targetSize));
output=normalize(output,'range',[-1 1]);
end
