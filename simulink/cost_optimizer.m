function best=cost_optimizer(parameters)
%COST_OPTIMIZER Grid-search processing nodes, staff, and bandwidth by cost/throughput.
if nargin<1, parameters=struct(); end
parameters=defaults(parameters,struct('arrivalRate',20,'serviceRate',12,'hoursPerYear',2000,'nodeCost',1200,'staffCost',8000,'bandwidthCost',100)); best=struct('cost',Inf);
for nodes=1:8
 for staff=1:8
  capacity=nodes*parameters.serviceRate; throughput=min(parameters.arrivalRate,capacity)*parameters.hoursPerYear;
  cost=nodes*parameters.nodeCost+staff*parameters.staffCost+parameters.bandwidthCost;
  if throughput>0 && cost/throughput<best.cost, best=struct('nodes',nodes,'staff',staff,'throughput',throughput,'annualCost',cost,'cost',cost/throughput); end
 end
end
end
function output=defaults(input,template), output=template; f=fieldnames(input); for k=1:numel(f),output.(f{k})=input.(f{k});end,end
