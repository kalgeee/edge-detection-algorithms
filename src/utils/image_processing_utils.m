function varargout = image_processing_utils(varargin)
% IMAGE_PROCESSING_UTILS Utility functions for edge detection algorithms
%
% This file contains utility functions used by the edge detection
% algorithms for image format conversion, validation, and preprocessing.
%
% Functions included:
%   - ensure_rgb_format: Convert any image format to RGB
%   - safe_rgb2gray: Safe RGB to grayscale conversion
%   - validate_image: Check if image is valid for processing
%
% Usage:
%   result = image_processing_utils('function_name', input_args);
%
% Author: Kalgee Joshi
% Date: September 2025

    if nargin == 0
        error('Please specify a function name to call');
    end
    
    function_name = varargin{1};
    args = varargin(2:end);
    
    switch lower(function_name)
        case 'ensure_rgb_format'
            varargout{1} = ensure_rgb_format(args{:});
        case 'safe_rgb2gray'
            varargout{1} = safe_rgb2gray(args{:});
        case 'validate_image'
            varargout{1} = validate_image(args{:});
        otherwise
            error('Unknown function: %s', function_name);
    end
end

function I_rgb = ensure_rgb_format(I)
% Convert any image format to RGB
    if ~isa(I, 'double')
        I = im2double(I);
    end
    
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
% Safe RGB to grayscale conversion
    if size(I, 3) == 3
        gray = 0.2989 * I(:,:,1) + 0.5870 * I(:,:,2) + 0.1140 * I(:,:,3);
    else
        gray = I(:,:,1);
    end
end

function is_valid = validate_image(I, min_size, max_size)
% Check if image is valid for processing
    if nargin < 2, min_size = [10, 10]; end
    if nargin < 3, max_size = [5000, 5000]; end
    
    is_valid = true;
    if ~isnumeric(I) || isempty(I)
        is_valid = false;
    end
    
    [h, w, ~] = size(I);
    if h < min_size(1) || w < min_size(2)
        is_valid = false;
    end
    if h > max_size(1) || w > max_size(2)
        is_valid = false;
    end
end
