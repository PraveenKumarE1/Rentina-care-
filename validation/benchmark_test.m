function metrics=benchmark_test(predicted,truth)
%BENCHMARK_TEST Calculate multiclass accuracy, referable sensitivity, specificity, and F1.
predicted=predicted(:);truth=truth(:); if numel(predicted)~=numel(truth),error('benchmark_test:Size','Inputs must match.');end
referable=truth>=2; predictedReferable=predicted>=2; tp=sum(referable & predictedReferable); tn=sum(~referable & ~predictedReferable); fp=sum(~referable & predictedReferable); fn=sum(referable & ~predictedReferable);
metrics=struct('accuracy',mean(predicted==truth),'sensitivity',tp/max(tp+fn,1),'specificity',tn/max(tn+fp,1),'f1',2*tp/max(2*tp+fp+fn,1),'n',numel(truth));
end
