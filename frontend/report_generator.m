function files = report_generator(result, patientId, outputDirectory, language)
%REPORT_GENERATOR Write a compact HTML report and optionally a PDF.
if nargin<4, language='en'; end
if nargin<3 || isempty(outputDirectory), outputDirectory='reports'; end
if ~exist(outputDirectory,'dir'), mkdir(outputDirectory); end
stamp=char(datetime('now','Format','yyyyMMdd_HHmmss')); base=fullfile(outputDirectory,['DR_' char(patientId) '_' stamp]);
labels=struct('en',struct('title','DR Screening Report','recommendation','Refer to ophthalmology'), ...
    'hi',struct('title','DR Screening Report','recommendation','Ophthalmology referral recommended'), ...
    'ta',struct('title','DR Screening Report','recommendation','Ophthalmology referral recommended'));
if ~isfield(labels,language), language='en'; end
html=sprintf(['<html><head><meta charset="utf-8"><title>%s</title></head><body>' ...
    '<h1>%s</h1><p>Patient: %s</p><p>Grade: %s (level %s)</p><p>Confidence: %.2f</p>' ...
    '<p>Recommendation: %s</p><p>Quality: %s</p></body></html>'],labels.(language).title,labels.(language).title,patientId,result.grading.label,num2str(result.grading.level),result.grading.confidence,ternary(result.grading.referable,labels.(language).recommendation,'Routine follow-up'),ternary(result.quality.passed,'Pass','Ungradable'));
htmlFile=[base '.html']; fid=fopen(htmlFile,'w'); fwrite(fid,html); fclose(fid); files=struct('html',htmlFile,'pdf','');
if exist('exportgraphics','file')
    fig=figure('Visible','off'); imshow(result.explainability.lesionOverlay); title(result.grading.label); exportgraphics(fig,[base '.pdf']); close(fig); files.pdf=[base '.pdf'];
end
end
function value=ternary(condition,a,b), if condition,value=a;else,value=b;end,end
