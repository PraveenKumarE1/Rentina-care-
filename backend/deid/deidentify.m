function [output, pseudonym] = deidentify(inputFile, outputFile, patientId)
%DEIDENTIFY Remove common DICOM identifiers and return a stable pseudonym.
arguments
    inputFile (1,1) string
    outputFile (1,1) string
    patientId (1,1) string
end
pseudonym = "DR-" + string(dec2hex(mod(sum(double(char(patientId)).*(1:numel(patientId))),2^31-1),8));
if endsWith(lower(inputFile),'.dcm')
    metadata=dicominfo(inputFile); fields={'PatientName','PatientID','PatientBirthDate','PatientAddress','InstitutionName'};
    for k=1:numel(fields), if isfield(metadata,fields{k}), metadata.(fields{k})=''; end, end
    metadata.PatientID=char(pseudonym); image=dicomread(inputFile); dicomwrite(image,outputFile,metadata,'CreateMode','Copy');
else
    copyfile(inputFile,outputFile); warning('deidentify:NonDICOM','Copied non-DICOM input; no metadata was present to strip.');
end
output=outputFile;
end
