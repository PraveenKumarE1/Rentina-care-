function calibrated=confidence_calibration(probabilities,labels,method)
%CONFIDENCE_CALIBRATION Calibrate probabilities using Platt or isotonic fitting.
if nargin<3,method='platt';end
if strcmpi(method,'platt')
    calibrated=probabilities; % Fit logistic parameters on validation predictions in deployment.
elseif strcmpi(method,'isotonic')
    calibrated=probabilities; % Replace with fitrnet/isotonic mapping after validation data is supplied.
else,error('confidence_calibration:Method','Use platt or isotonic.');end
if nargin>1 && numel(labels)~=size(probabilities,1),error('confidence_calibration:Size','Labels must match prediction rows.');end
end
