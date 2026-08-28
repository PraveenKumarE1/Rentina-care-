function output=simulation_runner(parameters)
%SIMULATION_RUNNER Simulate annual queue KPIs; mirrors telemedicine.slx signals.
if nargin<1,parameters=struct();end
parameters=defaults(parameters,struct('arrivalRate',20,'serviceRate',12,'nodes',2,'hours',2000));
capacity=parameters.serviceRate*parameters.nodes; output=struct('throughputPerYear',min(parameters.arrivalRate,capacity)*parameters.hours, ...
 'averageTurnaroundMinutes',60/max(capacity-parameters.arrivalRate,0.1),'utilisation',min(parameters.arrivalRate/capacity,1));
end
function output=defaults(input,template),output=template;f=fieldnames(input);for k=1:numel(f),output.(f{k})=input.(f{k});end,end
