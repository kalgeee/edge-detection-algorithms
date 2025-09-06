function edge_detection_demo()
% EDGE_DETECTION_DEMO Advanced edge detection algorithm demonstration
%
% This function demonstrates a multi-scale edge detection algorithm that
% combines Canny edge detection at different scales with morphological
% post-processing for robust edge detection in various image types.
%
% Features:
%   - Multi-scale Canny edge detection
%   - Automatic image format handling (RGB, grayscale, RGBA)
%   - Morphological noise reduction
%   - Comprehensive visualization and reporting
%   - Performance timing analysis
%
% Usage:
%   edge_detection_demo()
%
% Dependencies:
%   - MATLAB Image Processing Toolbox
%   - Test images in the current directory
%
% Author: [Your Name]
% Date: [Date]
% Version: 1.0

    % Display banner
    fprintf('=== Advanced Edge Detection Algorithm ===\n');
    fprintf('Multi-scale edge detection with morphological post-processing\n\n');
    
    % Test images (add your own image files here)
    test_images = {'zebra_01.jpg', 'woods.png'};
    
    % Process each test image
    for i = 1:length(test_images)
        if exist(test_images{i}, 'file')
            fprintf('Processing image %d: %s\n', i, test_images{i});
            process_image_edges(test_images{i});
        else
            fprintf('Image not found: %s (skipping)\n', test_images{i});
        end
    end
    
    fprintf('\nEdge detection processing complete.\n');
    fprintf('Results saved in individual output directories.\n');
end

function process_image_edges(image_path)
% PROCESS_IMAGE_EDGES Process a single image with edge detection
%
% Inputs:
%   image_path - String path to the input image file
%
% Processing steps:
%   1. Load and format image
%   2. Apply multi-scale edge detection
%   3. Create visualization
%   4. Save results and generate report

    try
        % Load and prepare image
        I_raw = imread(image_path);
        I = im2double(I_raw);
        
        % Ensure RGB format for consistent processing
        I = ensure_rgb_format(I);
        [h, w, ~] = size(I);
        
        fprintf('  Image dimensions: %dx%d\n', h, w);
        
        % Apply edge detection algorithm
        tic;
        edge_map = detect_refined_edges(I);
        processing_time = toc;
        
        fprintf('  Processing time: %.3f seconds\n', processing_time);
        
        % Create visualization
        create_edge_visualization(I, edge_map, image_path);
        
        % Save results
        save_edge_results(I, edge_map, image_path, processing_time);
        
        fprintf('  Results saved successfully\n\n');
        
    catch ME
        fprintf('  Error processing %s: %s\n', image_path, ME.message);
    end
end

function edge_map = detect_refined_edges(I)
% DETECT_REFINED_EDGES Main edge detection algorithm
%
% This function implements a multi-scale edge detection approach that
% combines Canny edge detection at multiple scales for robust edge
% detection across different image features.
%
% Algorithm steps:
%   1. Convert RGB to grayscale using luminance weights
%   2. Apply Canny edge detection at multiple scales (σ=1.0, 1.5)
%   3. Combine edge maps using logical OR
%   4. Apply morphological area opening to remove noise
%
% Inputs:
%   I - Input image (RGB double format)
%
% Outputs:
%   edge_map - Binary edge map (logical array)

    % Convert to grayscale for edge detection
    gray_img = safe_rgb2gray(I);
    
    % Multi-scale edge detection
    % Scale 1: Fine details (σ=1.0)
    edges1 = edge(gray_img, 'canny', [], 1.0);
    
    % Scale 2: Broader features (σ=1.5)
    edges2 = edge(gray_img, 'canny', [], 1.5);
    
    % Combine edges from different scales
    edge_map = edges1 | edges2;
    
    % Clean up edge map - remove small isolated edge fragments
    edge_map = bwareaopen(edge_map, 10);
end

function create_edge_visualization(I, edge_map, image_path)
% CREATE_EDGE_VISUALIZATION Create side-by-side visualization
%
% Creates a professional visualization showing the original image
% alongside the detected edge boundaries for comparison.

    [~, name, ~] = fileparts(image_path);
    
    % Create figure with appropriate size and layout
    figure('Name', sprintf('Edge Detection: %s', name), ...
           'Position', [100, 100, 1200, 500]);
    clf;
    
    % Original image (left panel)
    subplot(1, 2, 1);
    imshow(I);
    title('Original Image', 'FontSize', 16, 'FontWeight', 'bold');
    axis on;
    
    % Edge detection result (right panel)
    subplot(1, 2, 2);
    imshow(edge_map);
    title('Edge Boundaries', 'FontSize', 16, 'FontWeight', 'bold');
    axis on;
    
    % Add overall title
    sgtitle('Computer Vision: Edge Detection Analysis', ...
            'FontSize', 18, 'FontWeight', 'bold');
    
    % Force figure refresh
    drawnow;
end

function save_edge_results(I, edge_map, image_path, processing_time)
% SAVE_EDGE_RESULTS Save all results and generate report
%
% Saves the edge detection results, comparison images, and generates
% a comprehensive technical report of the processing.

    [~, name, ~] = fileparts(image_path);
    
    % Create output directory
    output_dir = sprintf('edge_results_%s', name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Save edge map
    imwrite(edge_map, fullfile(output_dir, 'edges.png'));
    
    % Save combined comparison image
    combined_image = create_combined_image(I, edge_map);
    imwrite(combined_image, fullfile(output_dir, 'comparison.png'));
    
    % Generate algorithm report
    create_algorithm_report(output_dir, image_path, processing_time);
    
    % Save current figure
    try
        saveas(gcf, fullfile(output_dir, 'visualization.png'));
    catch
        fprintf('  Warning: Could not save visualization figure\n');
    end
end

function combined_image = create_combined_image(original, edges)
% CREATE_COMBINED_IMAGE Create side-by-side combined image
%
% Creates a professional side-by-side comparison image suitable for
% portfolio presentation or technical documentation.

    [h, w, c] = size(original);
    
    % Convert edge map to RGB for concatenation
    if size(edges, 3) == 1
        edges_rgb = cat(3, edges, edges, edges);
    else
        edges_rgb = edges;
    end
    
    % Create combined image with white separator
    separator_width = 2;
    combined_image = zeros(h, w*2 + separator_width, c);
    
    % Place original image
    combined_image(:, 1:w, :) = original;
    
    % Add white separator
    combined_image(:, w+1:w+separator_width, :) = 1;
    
    % Place edge detection result
    combined_image(:, w+separator_width+1:end, :) = edges_rgb;
end

function create_algorithm_report(output_dir, image_path, processing_time)
% CREATE_ALGORITHM_REPORT Generate comprehensive technical report
%
% Creates a detailed technical report documenting the algorithm,
% parameters, and results for professional documentation.

    report_file = fullfile(output_dir, 'edge_detection_report.txt');
    fid = fopen(report_file, 'w');
    
    if fid == -1
        fprintf('  Warning: Could not create report file\n');
        return;
    end
    
    % Write comprehensive report
    fprintf(fid, 'Advanced Edge Detection Algorithm Report\n');
    fprintf(fid, '=======================================\n\n');
    fprintf(fid, 'Input Image: %s\n', image_path);
    fprintf(fid, 'Processing Date: %s\n', datestr(now));
    fprintf(fid, 'Processing Time: %.4f seconds\n\n', processing_time);
    
    fprintf(fid, 'Algorithm Description:\n');
    fprintf(fid, '---------------------\n');
    fprintf(fid, 'Multi-scale Canny edge detection with morphological post-processing\n\n');
    fprintf(fid, 'Processing Steps:\n');
    fprintf(fid, '1. RGB to grayscale conversion using standard luminance weights\n');
    fprintf(fid, '   Y = 0.2989*R + 0.5870*G + 0.1140*B\n');
    fprintf(fid, '2. Multi-scale Canny edge detection:\n');
    fprintf(fid, '   - Scale 1: σ = 1.0 (fine details)\n');
    fprintf(fid, '   - Scale 2: σ = 1.5 (broader features)\n');
    fprintf(fid, '3. Logical OR combination of edge maps\n');
    fprintf(fid, '4. Morphological area opening (minimum area: 10 pixels)\n\n');
    
    fprintf(fid, 'Technical Features:\n');
    fprintf(fid, '------------------\n');
    fprintf(fid, '- Multi-scale approach for robust edge detection\n');
    fprintf(fid, '- Handles various image formats (RGB, grayscale, RGBA)\n');
    fprintf(fid, '- Morphological post-processing for noise reduction\n');
    fprintf(fid, '- Preserves fine details while removing artifacts\n');
    fprintf(fid, '- Automatic threshold selection via Canny algorithm\n\n');
    
    fprintf(fid, 'Applications:\n');
    fprintf(fid, '------------\n');
    fprintf(fid, '- Object boundary detection and recognition\n');
    fprintf(fid, '- Image preprocessing for segmentation algorithms\n');
    fprintf(fid, '- Feature extraction for computer vision systems\n');
    fprintf(fid, '- Medical image analysis and diagnosis\n');
    fprintf(fid, '- Industrial quality control and inspection\n');
    fprintf(fid, '- Autonomous vehicle perception systems\n\n');
    
    fprintf(fid, 'Performance Characteristics:\n');
    fprintf(fid, '---------------------------\n');
    fprintf(fid, '- Computational complexity: O(n*m) where n,m are image dimensions\n');
    fprintf(fid, '- Memory usage: Linear with image size\n');
    fprintf(fid, '- Suitable for real-time applications on modern hardware\n');
    fprintf(fid, '- Scalable to high-resolution images\n');
    
    fclose(fid);
end

% Utility functions for image processing
function I_rgb = ensure_rgb_format(I)
% ENSURE_RGB_FORMAT Convert any image format to RGB
%
% Handles various input formats and converts them to consistent
% RGB format for uniform processing.

    if ndims(I) == 2
        % 2D grayscale to RGB
        I_rgb = cat(3, I, I, I);
    elseif size(I, 3) == 1
        % 3D single channel to RGB
        I_rgb = repmat(I, [1, 1, 3]);
    elseif size(I, 3) == 3
        % Already RGB
        I_rgb = I;
    elseif size(I, 3) == 4
        % RGBA - remove alpha channel
        I_rgb = I(:, :, 1:3);
    else
        error('Unsupported image format: %d channels', size(I, 3));
    end
end

function gray = safe_rgb2gray(I)
% SAFE_RGB2GRAY Safe RGB to grayscale conversion
%
% Performs RGB to grayscale conversion using standard luminance weights
% with fallback handling for various input formats.

    if size(I, 3) == 3
        % Standard RGB to grayscale conversion
        gray = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
    elseif size(I, 3) == 1
        % Already grayscale
        gray = I(:,:,1);
    else
        % Fallback: use first channel
        gray = I(:,:,1);
    end
end