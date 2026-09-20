function integration_tests()
%INTEGRATION_TESTS Check the public workflow and benchmark utility.
unit_tests(); metrics=benchmark_test([0 2 3 1],[0 2 1 1]); assert(metrics.n==4); assert(metrics.sensitivity>=0); disp('integration_tests: PASS');
end
