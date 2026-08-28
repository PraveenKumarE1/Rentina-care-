function correction=store_correction(screeningId,correctedGrade,reviewer)
%STORE_CORRECTION Return a correction record for persistence and later fine-tuning.
arguments
	screeningId (1,1) string
	correctedGrade (1,1) double {mustBeInteger,mustBeInRange(correctedGrade,0,4)}
	reviewer (1,1) string
end
correction=struct('screeningId',screeningId,'correctedGrade',correctedGrade,'reviewer',reviewer,'createdAt',datetime('now'),'status','queued-for-training');
end
