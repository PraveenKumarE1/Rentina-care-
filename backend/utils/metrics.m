function output=metrics(predicted,truth)
%METRICS Report classification metrics for benchmark experiments.
output=benchmark_test(predicted,truth);
output.confusionMatrix=confusionmat(truth,predicted,'Order',0:4);
end
