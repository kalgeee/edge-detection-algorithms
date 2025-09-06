function edge_comparison_demo()
% EDGE_COMPARISON_DEMO Comprehensive edge detection algorithm comparison
%
% This function creates an interactive comparison interface showing the
% performance and results of multiple edge detection algorithms on the
% same input image. Includes timing analysis and parameter visualization.
%
% Algorithms compared:
%   1. Canny Edge Detection
%   2. Sobel Operator
%   3. Laplacian Filter
%   4. Roberts Cross-Gradient
%   5. Prewitt Operator
%   6. Custom Multi-Scale Method
%
% Features:
%   - Side-by-side visual comparison
%   - Performance timing analysis
%   - Parameter display and documentation
%   - Comprehensive result saving
%
% Usage:
%   edge_comparison_demo()
%
% Author: [Your Name]
% Date: [Date]
% Version: 1.0

    % Display header
    fprintf('=== Edge Detection Algorithm Comparison ===\n\n');
    
    % Test images (add your image files here)
    test_images = {'zebra_01.jpg', 'indoor.jpg'};

    % Process each test image
    for i = 1:length(test_images)
        if exist(test_images{i}, 'file')
            fprintf('Processing image %d: %s\n', i, test_images{i});
            create_comparison_interface(test_images{i});
        else
            fprintf('Image not found: %s (skipping)\n', test_images{i});
        end
    end
    
    fprintf('\nAlgorithm comparison complete.\n');
end

function create_comparison_interface(image_path)
% CREATE_COMPARISON_INTERFACE Main comparison interface creation
%
% Creates a comprehensive 3x3 grid showing original image, six different
% edge detection algorithms, timing comparison, and parameter display.

    % Load and prepare image
    I_raw = imread(image_path);
    I = im2double(I_raw);
    I = ensure_rgb_format(I);
    gray_img = safe_rgb2gray(I);
    
    [~, name, ~] = fileparts(image_path);
    
    % Create main comparison figure
    fig = figure('Name', sprintf('Edge Detection Comparison: %s', name), ...
                 'Position', [50, 50, 1600, 1000]);
    
    % Set algorithm parameters
    params = get_default_parameters();
    
    % Process all algorithms with timing
    fprintf('  Running algorithm comparison...\n');
    [results, timings] = process_all_algorithms(gray_img, params);
    
    % Create the multi-panel layout
    create_algorithm_panels(I, results, timings, params, image_path);
    
    % Save comprehensive results
    save_comparison_results(I, results, timings, image_path);
    
    fprintf('  Comparison complete and saved\n\n');
end

function params = get_default_parameters()
% GET_DEFAULT_PARAMETERS Define default algorithm parameters
%
% Returns a structure containing optimized parameters for all algorithms.
% These parameters are tuned for general-purpose edge detection.

    params = struct();
    
    % Canny edge detection parameters
    params.canny_sigma = 1.0;           % Gaussian filter standard deviation
    params.canny_thresh_low = 0.1;      % Lower hysteresis threshold
    params.canny_thresh_high = 0.2;     % Upper hysteresis threshold
    
    % Sobel operator parameters
    params.sobel_threshold = 0.1;       % Gradient magnitude threshold
    
    % Laplacian filter parameters
    params.laplacian_alpha = 0.2;       % Laplacian kernel parameter
    
    % Roberts cross-gradient parameters
    params.roberts_threshold = 0.1;     % Gradient magnitude threshold
end

function [results, timings] = process_all_algorithms(gray_img, params)
% PROCESS_ALL_ALGORITHMS Apply all edge detection algorithms
%
% Processes the input image with all six edge detection algorithms
% and measures the processing time for each method.
%
% Inputs:
%   gray_img - Grayscale input image
%   params   - Algorithm parameters structure
%
% Outputs:
%   results  - Structure containing edge maps for each algorithm
%   timings  - Structure containing processing times

    results = struct();
    timings = struct();
    
    % 1. Canny Edge Detection
    tic;
    results.canny = edge(gray_img, 'canny', ...
        [params.canny_thresh_low, params.canny_thresh_high], ...
        params.canny_sigma);
    timings.canny = toc;
    
    % 2. Sobel Edge Detection
    tic;
    [Gx, Gy] = imgradientxy(gray_img, 'sobel');
    gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
    results.sobel = gradient_magnitude > params.sobel_threshold;
    timings.sobel = toc;
    
    % 3. Laplacian Edge Detection
    tic;
    laplacian_filter = fspecial('laplacian', params.laplacian_alpha);
    laplacian_response = imfilter(gray_img, laplacian_filter);
    results.laplacian = abs(laplacian_response) > params.roberts_threshold;
    timings.laplacian = toc;
    
    % 4. Roberts Edge Detection
    tic;
    roberts_h = [1 0; 0 -1];  % Roberts horizontal kernel
    roberts_v = [0 1; -1 0];  % Roberts vertical kernel
    roberts_x = imfilter(gray_img, roberts_h);
    roberts_y = imfilter(gray_img, roberts_v);
    roberts_magnitude = sqrt(roberts_x.^2 + roberts_y.^2);
    results.roberts = roberts_magnitude > params.roberts_threshold;
    timings.roberts = toc;
    
    % 5. Prewitt Edge Detection
    tic;
    results.prewitt = edge(gray_img, 'prewitt');
    timings.prewitt = toc;
    
    % 6. Custom Multi-Scale Method
    tic;
    results.custom = detect_custom_edges(gray_img);
    timings.custom = toc;
end

function edge_map = detect_custom_edges(gray_img)
% DETECT_CUSTOM_EDGES Custom multi-scale edge detection
%
% Implements the custom multi-scale edge detection algorithm that
% combines Canny edge detection at multiple scales.

    % Multi-scale edge detection
    edges1 = edge(gray_img, 'canny', [], 1.0);  % Fine scale
    edges2 = edge(gray_img, 'canny', [], 1.5);  % Coarse scale
    
    % Combine edges from different scales
    edge_map = edges1 | edges2;
    
    % Clean up edge map - remove small fragments
    edge_map = bwareaopen(edge_map, 10);
end

function create_algorithm_panels(I, results, timings, params, image_path)
% CREATE_ALGORITHM_PANELS Create the main multi-panel display
%
% Creates a 3x3 grid layout showing all algorithms, timing chart,
% and parameter information for comprehensive comparison.

    % Algorithm information
    algorithms = {'Original', 'Canny', 'Sobel', 'Laplacian', ...
                  'Roberts', 'Prewitt', 'Custom'};
    result_fields = {'', 'canny', 'sobel', 'laplacian', ...
                     'roberts', 'prewitt', 'custom'};
    
    % Create subplot layout (3x3 grid)
    for i = 1:length(algorithms)
        subplot(3, 3, i);
        
        if i == 1
            % Display original image
            imshow(I);
            title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');
        else
            % Display edge detection result with timing
            result_field = result_fields{i};
            imshow(results.(result_field));
            timing_ms = timings.(result_field) * 1000;
            title(sprintf('%s\n(%.2f ms)', algorithms{i}, timing_ms), ...
                  'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    
    % Add performance comparison chart (position 8)
    subplot(3, 3, 8);
    create_timing_chart(timings);
    
    % Add parameter information (position 9)
    subplot(3, 3, 9);
    display_algorithm_parameters(params);
    
    % Add overall title
    [~, name, ~] = fileparts(image_path);
    sgtitle(sprintf('Edge Detection Algorithm Comparison: %s', name), ...
            'FontSize', 16, 'FontWeight', 'bold');
end

function create_timing_chart(timings)
% CREATE_TIMING_CHART Create performance comparison bar chart
%
% Displays processing times for all algorithms with the fastest
% algorithm highlighted in green.

    % Extract algorithm names and times
    algorithms = {'Canny', 'Sobel', 'Laplacian', 'Roberts', 'Prewitt', 'Custom'};
    times_ms = [timings.canny, timings.sobel, timings.laplacian, ...
                timings.roberts, timings.prewitt, timings.custom] * 1000;
    
    % Create bar chart
    bars = bar(times_ms);
    set(gca, 'XTickLabel', algorithms);
    xtickangle(45);
    ylabel('Processing Time (ms)');
    title('Algorithm Performance', 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    % Highlight fastest algorithm
    [~, fastest_idx] = min(times_ms);
    bars.FaceColor = 'flat';
    bars.CData(fastest_idx,:) = [0 1 0];  % Green for fastest
end

function display_algorithm_parameters(params)
% DISPLAY_ALGORITHM_PARAMETERS Show current algorithm settings
%
% Creates a text display of all algorithm parameters for reference.

    axis off;
    
    % Format parameter text
    param_text = sprintf([...
        'Algorithm Parameters:\n\n' ...
        'Canny:\n' ...
        '  σ = %.1f\n' ...
        '  Low threshold = %.2f\n' ...
        '  High threshold = %.2f\n\n' ...
        'Sobel:\n' ...
        '  Threshold = %.2f\n\n' ...
        'Laplacian:\n' ...
        '  Alpha = %.2f\n\n' ...
        'Roberts:\n' ...
        '  Threshold = %.2f\n\n' ...
        'Note: Prewitt uses\n' ...
        'automatic thresholding'], ...
        params.canny_sigma, params.canny_thresh_low, ...
        params.canny_thresh_high, params.sobel_threshold, ...
        params.laplacian_alpha, params.roberts_threshold);
    
    % Display text
    text(0.05, 0.95, param_text, ...
         'FontSize', 10, ...
         'VerticalAlignment', 'top', ...
         'FontName', 'FixedWidth');
    title('Current Parameters', 'FontSize', 12, 'FontWeight', 'bold');
end

function save_comparison_results(I, results, timings, image_path)
% SAVE_COMPARISON_RESULTS Save all comparison results and reports
%
% Saves individual algorithm results, timing data, and generates
% comprehensive comparison documentation.

    [~, name, ~] = fileparts(image_path);
    output_dir = sprintf('edge_comparison_%s', name);
    
    % Create output directory
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Save individual algorithm results
    algorithms = fieldnames(results);
    for i = 1:length(algorithms)
        algorithm = algorithms{i};
        filename = sprintf('%s_edges.png', algorithm);
        imwrite(results.(algorithm), fullfile(output_dir, filename));
    end
    
    % Save timing data
    save(fullfile(output_dir, 'timing_results.mat'), 'timings');
    
    % Save comparison figure
    try
        saveas(gcf, fullfile(output_dir, 'complete_comparison.png'));
    catch ME
        fprintf('  Warning: Could not save figure: %s\n', ME.message);
    end
    
    % Generate comprehensive report
    create_comparison_report(output_dir, timings, image_path);
    
    fprintf('  Results saved to: %s\n', output_dir);
end

function create_comparison_report(output_dir, timings, image_path)
% CREATE_COMPARISON_REPORT Generate detailed comparison analysis
%
% Creates a comprehensive technical report comparing all algorithms
% with performance analysis and recommendations.

    report_file = fullfile(output_dir, 'algorithm_comparison_report.txt');
    fid = fopen(report_file, 'w');
    
    if fid == -1
        fprintf('  Warning: Could not create comparison report\n');
        return;
    end
    
    % Write comprehensive comparison report
    fprintf(fid, 'Edge Detection Algorithm Comparison Report\n');
    fprintf(fid, '=========================================\n\n');
    fprintf(fid, 'Input Image: %s\n', image_path);
    fprintf(fid, 'Analysis Date: %s\n\n', datestr(now));
    
    fprintf(fid, 'Algorithms Analyzed:\n');
    fprintf(fid, '-------------------\n');
    fprintf(fid, '1. Canny Edge Detection\n');
    fprintf(fid, '   - Optimal edge detection with hysteresis thresholding\n');
    fprintf(fid, '   - Best overall performance for most applications\n');
    fprintf(fid, '   - Excellent noise suppression\n\n');
    
    fprintf(fid, '2. Sobel Operator\n');
    fprintf(fid, '   - Gradient-based edge detection\n');
    fprintf(fid, '   - Fast processing suitable for real-time applications\n');
    fprintf(fid, '   - Good balance between speed and accuracy\n\n');
    
    fprintf(fid, '3. Laplacian Filter\n');
    fprintf(fid, '   - Second derivative edge detection\n');
    fprintf(fid, '   - High sensitivity to fine details and texture\n');
    fprintf(fid, '   - More susceptible to noise\n\n');
    
    fprintf(fid, '4. Roberts Cross-Gradient\n');
    fprintf(fid, '   - Simple and fast gradient operator\n');
    fprintf(fid, '   - Minimal computational overhead\n');
    fprintf(fid, '   - Good for simple edge detection tasks\n\n');
    
    fprintf(fid, '5. Prewitt Operator\n');
    fprintf(fid, '   - Gradient-based with smoothing properties\n');
    fprintf(fid, '   - Similar to Sobel with different kernel weights\n');
    fprintf(fid, '   - Balanced approach to noise handling\n\n');
    
    fprintf(fid, '6. Custom Multi-Scale Method\n');
    fprintf(fid, '   - Combines multiple Canny scales (σ=1.0, 1.5)\n');
    fprintf(fid, '   - Morphological post-processing for cleanup\n');
    fprintf(fid, '   - Robust performance on complex images\n\n');
    
    % Performance analysis
    fprintf(fid, 'Performance Results:\n');
    fprintf(fid, '-------------------\n');
    
    algorithms = fieldnames(timings);
    times_ms = structfun(@(x) x * 1000, timings);
    [sorted_times, sort_idx] = sort(times_ms);
    sorted_algorithms = algorithms(sort_idx);
    
    fprintf(fid, 'Processing Time Ranking (fastest to slowest):\n');
    for i = 1:length(sorted_algorithms)
        fprintf(fid, '%d. %s: %.3f ms\n', i, upper(sorted_algorithms{i}), sorted_times(i));
    end
    
    fprintf(fid, '\nPerformance Analysis:\n');
    fprintf(fid, '- Fastest algorithm: %s (%.3f ms)\n', upper(sorted_algorithms{1}), sorted_times(1));
    fprintf(fid, '- Slowest algorithm: %s (%.3f ms)\n', upper(sorted_algorithms{end}), sorted_times(end));
    fprintf(fid, '- Speed ratio: %.1fx difference\n', sorted_times(end)/sorted_times(1));
    
    % Recommendations
    fprintf(fid, '\nRecommendations by Use Case:\n');
    fprintf(fid, '---------------------------\n');
    fprintf(fid, 'Real-time applications: Roberts or Sobel (fastest processing)\n');
    fprintf(fid, 'High-quality edge detection: Canny (optimal results)\n');
    fprintf(fid, 'Fine detail preservation: Laplacian or Custom multi-scale\n');
    fprintf(fid, 'Noise-robust processing: Custom multi-scale or Canny\n');
    fprintf(fid, 'General-purpose applications: Canny or Prewitt\n');
    fprintf(fid, 'Resource-constrained systems: Roberts or Sobel\n\n');
    
    fprintf(fid, 'Technical Specifications:\n');
    fprintf(fid, '------------------------\n');
    fprintf(fid, 'All algorithms tested on grayscale converted image\n');
    fprintf(fid, 'Timing measured on MATLAB environment\n');
    fprintf(fid, 'Results may vary based on hardware and image complexity\n');
    fprintf(fid, 'Parameters optimized for general-purpose edge detection\n');
    
    fclose(fid);
end

% Utility functions
function I_rgb = ensure_rgb_format(I)
% ENSURE_RGB_FORMAT Convert image to RGB format
    if ndims(I) == 2
        I_rgb = cat(3, I, I, I);
    elseif size(I, 3) == 1
        I_rgb = repmat(I, [1, 1, 3]);
    elseif size(I, 3) == 3
        I_rgb = I;
    elseif size(I, 3) == 4
        I_rgb = I(:, :, 1:3);
    else
        error('Unsupported image format: %d channels', size(I, 3));
    end
end

function gray = safe_rgb2gray(I)
% SAFE_RGB2GRAY Convert RGB to grayscale safely
    if size(I, 3) == 3
        gray = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
    else
        gray = I(:,:,1);
    end
end