function overlay = lesion_overlay(image, segmentation)
%LESION_OVERLAY Render colour-coded lesion evidence on the source image.
overlay=im2double(image); if size(overlay,3)==1, overlay=repmat(overlay,1,1,3); end
masks={segmentation.microaneurysms.mask,segmentation.exudates.mask,segmentation.haemorrhages.mask,segmentation.neovascularisation.mask};
colors={[1 0 1],[1 1 0],[1 0 0],[0 1 1]};
for k=1:numel(masks), for c=1:3, channel=overlay(:,:,c); channel(masks{k})=0.45*channel(masks{k})+0.55*colors{k}(c); overlay(:,:,c)=channel; end, end
overlay=im2uint8(min(max(overlay,0),1));
end
