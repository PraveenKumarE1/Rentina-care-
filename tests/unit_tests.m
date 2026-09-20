function unit_tests()
%UNIT_TESTS Smoke tests using synthetic retinal-like data.
img=uint8(zeros(256,256,3)); [x,y]=meshgrid(1:256,1:256); mask=(x-128).^2+(y-128).^2<100^2; for c=1:3, img(:,:,c)=uint8(mask)*120; end
result=main_pipeline(img); assert(isfield(result,'grading')); assert(numel(result.grading.probabilities)==5); assert(isfield(result,'quality')); disp('unit_tests: PASS');
end
