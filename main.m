function main()
%MAIN Command-line entry point for automated DR screening.
startup();
fprintf('\nDR Screening System\n');
fprintf('1. Screen an image\n2. Run synthetic smoke test\n3. Exit\n');
choice = input('Select an option: ');
switch choice
    case 1
        filename = input('Image path: ','s');
        result = main_pipeline(filename);
        disp(result.summary);
    case 2
        unit_tests();
    otherwise
        fprintf('Exit.\n');
end
end
