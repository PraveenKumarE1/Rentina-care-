function report=clinical_validation(predicted,truth)
%CLINICAL_VALIDATION Summarize a locked test-set evaluation.
report=benchmark_test(predicted,truth); report.targetReferableSensitivity=0.90; report.targetReferableSpecificity=0.85; report.meetsTarget=report.sensitivity>=report.targetReferableSensitivity && report.specificity>=report.targetReferableSpecificity;
end
