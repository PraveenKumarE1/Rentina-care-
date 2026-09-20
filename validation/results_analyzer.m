function summary=results_analyzer(results)
%RESULTS_ANALYZER Aggregate benchmark result structs from multiple datasets.
if isempty(results),summary=struct();return;end
fields={'accuracy','sensitivity','specificity','f1'};summary=struct();for k=1:numel(fields),summary.(fields{k})=mean([results.(fields{k})]);end
end
