function segmentation = segmentation(inputImage, quality)
%SEGMENTATION Produce interpretable classical proposals for retinal structures.
if size(inputImage,3)==3, rgb=im2double(inputImage); green=rgb(:,:,2); else, green=im2double(inputImage); rgb=repmat(green,1,1,3); end
green=mat2gray(green); bright=mat2gray(rgb(:,:,1)+rgb(:,:,2)+rgb(:,:,3));
opticDisc = imclose(bright > prctile(bright(:),99),strel('disk',8));
opticDisc = bwareafilt(opticDisc,1);
exudates = bwareaopen(imopen(bright > prctile(bright(:),98),strel('disk',2)),4);
exudates(opticDisc)=false;
dark = mat2gray(1-green);
hemorrhages = bwareaopen(imopen(dark > prctile(dark(:),97),strel('disk',2)),5);
microaneurysms = bwareaopen(imopen(dark > prctile(dark(:),99),strel('disk',1)),2);
microaneurysms(opticDisc)=false;
vessels = imbinarize(imadjust(1-green),'adaptive','Sensitivity',0.55);
vessels = bwareaopen(vessels,5); vessels(opticDisc)=false;
props = @(mask) struct('mask',mask,'count',nnz(mask),'area',nnz(mask)/numel(mask));
segmentation = struct('opticDisc',props(opticDisc),'fovea',struct('mask',false(size(green)),'count',0,'area',0), ...
    'vessels',props(vessels),'microaneurysms',props(microaneurysms), ...
    'exudates',props(exudates),'haemorrhages',props(hemorrhages), ...
    'neovascularisation',struct('mask',false(size(green)),'count',0,'area',0));
segmentation.features = struct('vesselDensity',segmentation.vessels.area, ...
    'tortuosity',NaN,'cupToDiscRatio',NaN);
if quality.passed == false, segmentation.warning = 'Interpret results only after a gradable acquisition.'; end
end
