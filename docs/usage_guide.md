# Usage Guide

## Prerequisites

- MATLAB R2018b or later
- Image Processing Toolbox
- Test images in the `examples/` folder

## Quick Start

### 1. Running Single Algorithm Demo

```matlab
% Navigate to the project directory
cd path/to/edge-detection-algorithms

% Add source directory to MATLAB path
addpath('src')

% Run the main demo
edge_detection_demo
```

This will process all test images with the custom multi-scale algorithm and create:
- Side-by-side visualization windows
- Results saved in `edge_results_[imagename]/` folders

### 2. Running Algorithm Comparison

```matlab
% Add source directory to MATLAB path
addpath('src')

% Run comparison interface
edge_comparison_demo
```

This creates a comprehensive 3x3 comparison showing:
- Original image
- Results from 6 different algorithms
- Performance timing chart
- Algorithm parameters display

## Detailed Usage

### Adding Your Own Test Images

1. **Place images in the examples folder:**
   ```
   examples/
   ├── your_image1.jpg
   ├── your_image2.png
   └── zebra_01.jpg
   ```

2. **Update the test image list in the scripts:**
   
   Edit `src/edge_detection_demo.m`:
   ```matlab
   test_images = {'zebra_01.jpg', 'your_image1.jpg', 'your_image2.png'};
   ```
   
   Edit `src/edge_comparison_demo.m`:
   ```matlab
   test_images = {'zebra_01.jpg', 'your_image1.jpg', 'your_image2.png'};
   ```

### Supported Image Formats

- **File types**: JPEG (.jpg, .jpeg), PNG (.png), TIFF (.tif, .tiff), BMP (.bmp)
- **Color formats**: RGB, Grayscale, RGBA (alpha channel will be removed)
- **Recommended size**: 500×500 to 1000×1000 pixels
- **Size limits**: Minimum 50×50, Maximum 5000×5000 pixels

### Understanding the Output

#### Single Algorithm Results (`edge_results_[imagename]/`)
- `edges.png` - Binary edge map
- `comparison.png` - Side-by-side original and edges
- `visualization.png` - MATLAB figure output
- `edge_detection_report.txt` - Technical analysis report

#### Comparison Results (`edge_comparison_[imagename]/`)
- `[algorithm]_edges.png` - Individual algorithm results
- `complete_comparison.png` - Full 3×3 comparison grid
- `timing_results.mat` - Performance timing data
- `algorithm_comparison_report.txt` - Comprehensive analysis

## Customizing Algorithm Parameters

### Editing Parameters

Open `src/edge_comparison_demo.m` and modify the `get_default_parameters()` function:

```matlab
function params = get_default_parameters()
    params = struct();
    
    % Canny edge detection parameters
    params.canny_sigma = 1.0;           % Gaussian filter std deviation
    params.canny_thresh_low = 0.1;      % Lower hysteresis threshold  
    params.canny_thresh_high = 0.2;     % Upper hysteresis threshold
    
    % Sobel operator parameters
    params.sobel_threshold = 0.1;       % Gradient magnitude threshold
    
    % Laplacian filter parameters
    params.laplacian_alpha = 0.2;       % Laplacian kernel parameter
    
    % Roberts cross-gradient parameters
    params.roberts_threshold = 0.1;     % Gradient magnitude threshold
end
```

### Parameter Effects

- **Higher sigma**: Smoother edges, less noise, fewer details
- **Lower sigma**: More detailed edges, more noise
- **Higher thresholds**: Fewer edges detected
- **Lower thresholds**: More edges detected, more noise

## Advanced Usage

### Processing Single Images

```matlab
% Load and process a specific image
I = imread('examples/zebra_01.jpg');
I = im2double(I);

% Apply custom edge detection
edges = detect_refined_edges(I);

% Display results
figure;
subplot(1,2,1); imshow(I); title('Original');
subplot(1,2,2); imshow(edges); title('Edges');
```

### Using Utility Functions

```matlab
% Add utilities to path
addpath('src/utils')

% Convert image format
I_rgb = image_processing_utils('ensure_rgb_format', I);

% Convert to grayscale safely
gray = image_processing_utils('safe_rgb2gray', I_rgb);

% Validate image
is_valid = image_processing_utils('validate_image', I);
```

## Troubleshooting

### Common Issues

**Error: "File not found"**
- Check that image files are in the `examples/` folder
- Verify filename spelling and case sensitivity
- Ensure file extensions are included in the `test_images` array

**Error: "Out of memory"**
- Use smaller images (resize to 800×600 or smaller)
- Close other MATLAB applications
- Clear workspace: `clear; clc;`

**Error: "Function not found"**
- Ensure you've added the source directory to MATLAB path:
  ```matlab
  addpath('src')
  addpath('src/utils')
  ```

**Figures not displaying properly**
- Check MATLAB figure settings
- Try: `set(0,'DefaultFigureVisible','on')`
- Close existing figures: `close all`

### Performance Optimization

**For faster processing:**
- Use JPEG format (faster to load than PNG)
- Resize large images before processing
- Process images individually rather than in batch
- Close figure windows between runs

**For better results:**
- Use high-contrast images
- Ensure good lighting in photographs
- Avoid heavily compressed images
- Use images with clear object boundaries

### Memory Management

```matlab
% Clear workspace between runs
clear; clc; close all;

% Monitor memory usage
memory  % (Windows only)

% Force garbage collection
pack
```

## Example Workflow

```matlab
% Complete workflow example
clear; clc; close all;

% 1. Set up environment
addpath('src');
addpath('src/utils');

% 2. Run single algorithm demo
edge_detection_demo;

% 3. Run comparison analysis  
edge_comparison_demo;

% 4. Check results
ls results/
```

This workflow will process all test images and generate comprehensive results for portfolio demonstration.
