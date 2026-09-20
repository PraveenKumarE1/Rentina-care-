function grading = grading(segmentation, quality, varargin)
%GRADING Assign International Clinical DR Scale level 0-4 from lesion proposals.
if ~quality.passed
    grading = struct('level',NaN,'label','Ungradable','confidence',0,'referable',false,'probabilities',NaN(1,5)); return
end
ma = segmentation.microaneurysms.count; ex = segmentation.exudates.count; he = segmentation.haemorrhages.count; nv = segmentation.neovascularisation.count;
if nv > 0, level=4; elseif he >= 20, level=3; elseif he > 0 || ex > 0, level=2; elseif ma > 0, level=1; else, level=0; end
labels = {'No DR','Mild NPDR','Moderate NPDR','Severe NPDR','PDR'};
probabilities = zeros(1,5); probabilities(level+1)=1;
gradeEvidence = struct('microaneurysms',ma,'exudates',ex,'haemorrhages',he,'neovascularisation',nv);
grading=struct('level',level,'label',labels{level+1},'confidence',0.55+0.1*(level==0), ...
    'referable',level>=2,'probabilities',probabilities,'evidence',gradeEvidence, ...
    'method','interpretable rule-based baseline; configure trained ensemble weights for clinical use');
end
