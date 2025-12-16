#!/usr/bin/env python3
"""
Generate plot matrices for each parameterization group.

Creates a figure with:
- 4 columns (one for each age group: p3, p12, p20, p60)
- Multiple rows with specific plot types:
  - Row 1: _num_targets_pie.svg for each age group
  - Row 2: _green_white_cluster_heatmap.svg for each age group
  - Row 3: _effect_significance.svg for each age group
  - Row 4: blueyellow_probability_heatmap.svg for each age group
  - Row 5+: Additional output files from process and helper scripts

Usage:
    python generate_figure_from_outputs.py [--parameterization PARAM] [--output_dir OUTPUT_DIR]
    
    If no arguments provided, generates figures for all parameterizations.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib
# Configure matplotlib for maximum PDF quality
# Note: Some PDF compression settings are not available in all matplotlib versions
try:
    # Try to set compression settings if available
    if hasattr(matplotlib.backends.backend_pdf, 'PdfPages'):
        pass  # PDF backend will use default settings
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Try to import image handling libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not available. Some image formats may not be supported.")

# Base directory
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_BASE = REPO_ROOT / "02_output"
FIGURE_OUTPUT_DIR = REPO_ROOT / "figure_generation" / "generated_figures"

# Age groups in order (columns)
AGE_GROUPS = ['p3', 'p12', 'p20', 'p60']

# Plot types to include (rows) - in order
# Each entry: (pattern, description, search_dirs)
PLOT_ROWS = [
    # Row 1: num_targets_pie
    ('*_ALL_*_num_targets_pie', 'Number of Targets Pie Chart', ['analysis']),
    # Row 2: green_white_cluster_heatmap
    ('*_ALL_*_green_white_cluster_heatmap', 'Green/White Cluster Heatmap', ['analysis']),
    # Row 3: effect_significance
    ('*_ALL_*_filters_effect_significance', 'Effect Significance (Volcano)', ['analysis']),
    # Row 4: blueyellow_probability_heatmap
    ('*_ALL_*_blueyellow_probability_heatmap', 'Blue/Yellow Probability Heatmap', ['analysis']),
    # Row 5+: Additional main processing plots
    ('*_ALL_*_cluster_diagnostics', 'Cluster Diagnostics', ['analysis']),
    ('*_ALL_*_kmeans', 'K-Means Clustering', ['analysis']),
    ('*_ALL_*_tsne', 't-SNE Visualization', ['analysis']),
    ('*_ALL_*_Hanstyle_cluster_heatmap', 'Han-style Cluster Heatmap', ['analysis']),
    ('*_ALL_*_ExtendedDataFig10_Recreation', 'Extended Data Fig 10 Recreation', ['analysis']),
    ('*_ALL_*_upsetplot', 'Motif Over/Under Representation (UpsetPlot)', ['analysis']),
    ('*_ALL_*_per_cell_proj_strength', 'Per-Cell Projection Strength', ['analysis']),
    # Plots saved directly to parameterization directory (not in analysis/)
    ('*_ALL_*_Region_Probabilities', 'Region Probabilities', ['']),
    ('*_ALL_*_Roots', 'Roots', ['']),
    ('*_ALL_*_Simplified_Pi', 'Simplified Pi', ['']),
    ('*_ALL_*_Calculated_Value', 'Calculated Value', ['']),
    # Helper script outputs (cross-age summaries - shown once per parameterization, not per age)
    ('UMI_composition_by_age', 'UMI Composition by Age', ['helpers/03_composition']),
    ('ProjCount_composition_by_age', 'ProjCount Composition by Age', ['helpers/03_composition']),
    ('proportion_plot', 'Proportions Over Time (Stacked Bar)', ['helpers/04_proportions_over_time_stats']),
    ('proportion_line_plot', 'Proportions Over Time (Line Plot)', ['helpers/04_proportions_over_time_stats']),
    ('clr_pca_plot', 'CLR PCA Plot', ['helpers/04_proportions_over_time_stats']),
    ('chi_square_residuals_heatmap', 'Chi-Square Residuals Heatmap', ['helpers/04_proportions_over_time_stats']),
]


def find_plot_file(param_dir, age, pattern, search_dirs):
    """
    Find a plot file matching the pattern for a specific age group.
    
    Args:
        param_dir: Path to parameterization directory for this age
        age: Age group (e.g., 'p3', 'p12')
        pattern: File pattern to match (e.g., '*_ALL_*_num_targets_pie')
        search_dirs: List of subdirectories to search
    
    Returns:
        Path to found file (preferring .svg, then .png, then .pdf) or None
    """
    if not param_dir.exists():
        return None
    
    # Determine if this is a model-specific plot that should be in uniform/ or region_specific/ subdirectories
    is_model_specific = any(keyword in pattern.lower() for keyword in ['upsetplot', 'effect_significance', 'per_cell_proj_strength'])
    
    # Try to match age in pattern (case-insensitive)
    # Pattern might be like '*_ALL_*_num_targets_pie' and we need to find files like 'p3_ALL_*_num_targets_pie.svg'
    # For patterns with *_ALL_*, replace with age_ALL_* and keep the wildcard for filter type
    if '*_ALL_*' in pattern:
        # Replace *_ALL_* with age_ALL_* (keeping the * for filter type like "minimal_filters")
        age_pattern = pattern.replace('*_ALL_*', f'{age}_ALL_*')
    else:
        # For patterns without *_ALL_*, replace * with age_*
        age_pattern = pattern.replace('*', f'{age}_*')
    
    # For model-specific plots, search in uniform/ and region_specific/ subdirectories
    # For backward compatibility, also check main analysis/ directory
    search_locations = []
    if is_model_specific:
        # Model-specific: search in uniform/ and region_specific/ subdirectories first
        for model_type in ['uniform', 'region_specific']:
            for search_dir_str in search_dirs:
                if search_dir_str:
                    search_locations.append((param_dir / search_dir_str / model_type, model_type))
                else:
                    search_locations.append((param_dir / 'analysis' / model_type, model_type))
        # Also check main directory for backward compatibility
        for search_dir_str in search_dirs:
            if search_dir_str:
                search_locations.append((param_dir / search_dir_str, None))
            else:
                search_locations.append((param_dir / 'analysis', None))
    else:
        # Non-model-specific: search in main directories only
        for search_dir_str in search_dirs:
            if search_dir_str:
                search_locations.append((param_dir / search_dir_str, None))
            else:
                search_locations.append((param_dir, None))
    
    for search_dir, model_type in search_locations:
        if not search_dir.exists():
            continue
        
        # For model-specific files, add model suffix to pattern
        if is_model_specific and model_type:
            # Add model suffix to pattern (e.g., *upsetplot -> *upsetplot_uniform or *upsetplot_region_specific)
            base_pattern = age_pattern.replace('*_ALL_*', f'{age}_ALL_*') if '*_ALL_*' in age_pattern else age_pattern
            model_pattern = base_pattern.replace('*upsetplot', f'*upsetplot_{model_type}').replace('*effect_significance', f'*effect_significance_{model_type}').replace('*per_cell_proj_strength', f'*per_cell_proj_strength_{model_type}')
            patterns_to_try = [
                model_pattern,
                base_pattern,  # Also try without model suffix for backward compatibility
                age_pattern,
                pattern.replace('*_ALL_*', f'{age.upper()}_ALL_*') if '*_ALL_*' in pattern else pattern.replace('*', f'{age.upper()}_*'),
                pattern.replace('*_ALL_*', f'{age.lower()}_ALL_*') if '*_ALL_*' in pattern else pattern.replace('*', f'{age.lower()}_*'),
                pattern,  # Original pattern
            ]
        else:
            # Non-model-specific: use standard patterns
            patterns_to_try = [
                age_pattern,
                pattern.replace('*_ALL_*', f'{age.upper()}_ALL_*') if '*_ALL_*' in pattern else pattern.replace('*', f'{age.upper()}_*'),
                pattern.replace('*_ALL_*', f'{age.lower()}_ALL_*') if '*_ALL_*' in pattern else pattern.replace('*', f'{age.lower()}_*'),
                pattern,  # Original pattern
            ]
        
        # Prefer PNG files first (they embed better in PDFs)
        # Try all PNG patterns first
        for pat in patterns_to_try:
            png_matches = list(search_dir.glob(f'{pat}.png'))
            if png_matches:
                return png_matches[0]
            
        # Then try PDF files (across all patterns)
        for pat in patterns_to_try:
            pdf_matches = list(search_dir.glob(f'{pat}.pdf'))
            if pdf_matches:
                return pdf_matches[0]
            
        # SVG last (requires special handling)
        for pat in patterns_to_try:
            svg_matches = list(search_dir.glob(f'{pat}.svg'))
            if svg_matches:
                return svg_matches[0]
    
    return None


def load_image_safe(image_path):
    """
    Load an image file, handling different formats.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image array or None if loading fails
    """
    if not image_path or not image_path.exists():
        return None
    
    # Skip SVG files - matplotlib cannot load them directly
    if image_path.suffix.lower() == '.svg':
        # Try to find PNG or PDF alternative
        png_path = image_path.with_suffix('.png')
        pdf_path = image_path.with_suffix('.pdf')
        
        if png_path.exists():
            image_path = png_path
        elif pdf_path.exists():
            image_path = pdf_path
        else:
            # No alternative found, try to use svglib if available
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPM
                import io
                
                # Convert SVG to PNG in memory with high DPI for quality
                drawing = svg2rlg(str(image_path))
                if drawing:
                    img_data = renderPM.drawToString(drawing, fmt='PNG', dpi=600)
                    img = Image.open(io.BytesIO(img_data))
                    return np.array(img)
                else:
                    # Silently return None - will show "Not Available" in figure
                    return None
            except ImportError:
                # Silently return None - will show "Not Available" in figure
                # Only print warning for debug if needed
                return None
            except Exception as e:
                # Silently return None - will show "Not Available" in figure
                return None
    
    try:
        # Try matplotlib first (works for PNG, PDF, etc.)
        # Use PIL for better quality control if available
        if HAS_PIL:
            # PIL preserves image quality better than mpimg
            pil_img = Image.open(image_path)
            # Convert to RGB if necessary (handles RGBA, etc.)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img = np.array(pil_img)
            return img
        else:
            img = mpimg.imread(str(image_path))
            return img
    except Exception as e1:
        if HAS_PIL:
            try:
                # Try PIL for other formats
                img = Image.open(image_path)
                return np.array(img)
            except Exception as e2:
                print(f"    Warning: Could not load {image_path}: {e1}, {e2}")
                return None
        else:
            print(f"    Warning: Could not load {image_path}: {e1}")
            return None


def create_plot_matrix(param_name, age_param_dirs, output_path):
    """
    Create a plot matrix figure for a parameterization.
    
    Args:
        param_name: Name of parameterization (e.g., "01.minimal_filter_parameters_i1_r1_t1_u2")
        age_param_dirs: Dictionary mapping age -> param_dir Path
        output_path: Where to save the figure
    
    Returns:
        Path to saved figure or None
    """
    # Find the shared helpers directory for this parameterization
    # It should be at: 02_output/{param_name}_helpers/
    helpers_base = OUTPUT_BASE / f"{param_name}_helpers"
    
    # Determine which rows we can populate
    available_rows = []
    plot_data = {}  # {row_idx: {age: plot_path}}
    
    for row_idx, (pattern, description, search_dirs) in enumerate(PLOT_ROWS):
        row_plots = {}
        row_has_data = False
        
        # Check if this is a helper script output (cross-age summary)
        is_helper_output = any('helpers' in str(sd) for sd in search_dirs)
        
        if is_helper_output:
            # For helper outputs, search in the shared helpers directory
            # They're the same across all ages, so find once and use for all
            helper_plot_path = None
            for search_dir_str in search_dirs:
                # Remove 'helpers/' prefix if present
                rel_path = search_dir_str.replace('helpers/', '')
                helper_dir = helpers_base / rel_path if rel_path else helpers_base
                
                if helper_dir.exists():
                    # Try to find the file (no age prefix needed for helper outputs)
                    for ext in ['png', 'pdf', 'svg']:
                        matches = list(helper_dir.glob(f'{pattern}.{ext}'))
                        if matches:
                            helper_plot_path = matches[0]
                            break
                    if helper_plot_path:
                        break
            
            if helper_plot_path:
                # Use the same plot for all age columns
                for age in AGE_GROUPS:
                    if age in age_param_dirs:
                        row_plots[age] = helper_plot_path
                        row_has_data = True
        else:
            # For per-age plots, search in each age's directory
            for age in AGE_GROUPS:
                if age not in age_param_dirs:
                    continue
                
                param_dir = age_param_dirs[age]
                plot_path = find_plot_file(param_dir, age, pattern, search_dirs)
                
                if plot_path:
                    row_plots[age] = plot_path
                    row_has_data = True
        
        if row_has_data:
            available_rows.append((row_idx, description, row_plots))
            plot_data[row_idx] = row_plots
    
    if not available_rows:
        print(f"  Warning: No plots found for {param_name}")
        return None
    
    # Create figure
    n_rows = len(available_rows)
    n_cols = len(AGE_GROUPS)
    
    # Calculate figure size - maximize size while staying within pixel limits
    # Use larger subplots for more pixels, but calculate DPI to fit limits
    # Use 6x5 inches per subplot for maximum detail
    fig_width = n_cols * 6
    fig_height = n_rows * 5
    
    # For TIFF, calculate maximum DPI that fits within matplotlib's limits (65536 pixels max)
    # We want to maximize both size AND DPI for best quality
    # Push to the absolute limit for maximum quality
    max_pixels = 65535  # At the 2^16 limit (65536) for maximum quality
    max_dpi_width = max_pixels / fig_width if fig_width > 0 else 600
    max_dpi_height = max_pixels / fig_height if fig_height > 0 else 600
    # Use the smaller of the two to ensure both dimensions fit
    # This gives us the maximum DPI we can use
    target_dpi = min(int(max_dpi_width), int(max_dpi_height))
    # Ensure we're using a reasonable DPI (at least 400)
    target_dpi = max(target_dpi, 400)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.suptitle(f'{param_name}', fontsize=16, fontweight='bold', y=0.995)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Populate the matrix
    for plot_row_idx, (orig_row_idx, description, row_plots) in enumerate(available_rows):
        for col_idx, age in enumerate(AGE_GROUPS):
            ax = axes[plot_row_idx, col_idx]
            
            if age in row_plots:
                plot_path = row_plots[age]
                img = load_image_safe(plot_path)
                
                if img is not None:
                    # Display image - let it fill the axes naturally
                    # Use 'nearest' interpolation to avoid blurring
                    ax.imshow(img, interpolation='nearest', aspect='auto', origin='upper')
                    ax.axis('off')
                    # Remove all margins to maximize image area
                    ax.margins(0, 0)
                else:
                    ax.text(0.5, 0.5, 'Image\nLoad Error', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Not\nAvailable', 
                       ha='center', va='center', transform=ax.transAxes,
                       color='gray', fontsize=10)
                ax.axis('off')
            
            # Set column headers (age labels) on first row
            if plot_row_idx == 0:
                ax.set_title(age.upper(), fontsize=12, fontweight='bold', pad=5)
            
            # Set row labels (plot type) on first column
            if col_idx == 0:
                ax.text(-0.1, 0.5, description, 
                       ha='right', va='center', transform=ax.transAxes,
                       fontsize=10, rotation=90)
    
    # Use tight layout but allow more space for images
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98], pad=0.5)
    
    # Save figure with maximum quality settings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine output format based on file extension
    output_format = output_path.suffix.lower()
    
    # For TIFF, save directly at high DPI (lossless format)
    if output_format == '.tiff' or output_format == '.tif':
        print(f"    Saving as TIFF at DPI: {target_dpi} (figure size: {fig_width:.1f}\" x {fig_height:.1f}\")")
        # Save as PNG to temporary file first, then convert to TIFF
        # This ensures we preserve full resolution
        if HAS_PIL:
            # Disable PIL's decompression bomb check for large scientific images
            Image.MAX_IMAGE_PIXELS = None
            # Save to temporary PNG file (not buffer) to preserve full resolution
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
                tmp_png_path = tmp_png.name
            try:
                # Save figure to PNG at full DPI
                fig.savefig(tmp_png_path, dpi=target_dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none',
                           format='png', pad_inches=0.1)
                # Load with PIL and save as TIFF with lossless compression
                img = Image.open(tmp_png_path)
                # Keep original mode (might be RGBA) for maximum quality
                # If it's RGBA, that's 4 channels instead of 3, which increases file size
                # Convert to RGB only if necessary for compatibility
                if img.mode == 'RGBA':
                    # Keep RGBA for maximum quality (4 channels = larger file)
                    pass
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save as TIFF with no compression for maximum quality
                # Use 'none' compression for maximum quality
                img.save(output_path, 'TIFF', compression='none', dpi=(target_dpi, target_dpi))
                # Clean up temp file
                import os
                os.unlink(tmp_png_path)
            except Exception as e:
                # Clean up temp file on error
                import os
                if os.path.exists(tmp_png_path):
                    os.unlink(tmp_png_path)
                raise e
        else:
            # Fallback to matplotlib's TIFF if PIL not available
            fig.savefig(output_path, dpi=target_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       format='tif', pad_inches=0.1)
    elif output_format == '.pdf':
        # For PDF, save as high-res PNG first, then convert
        png_path = output_path.with_suffix('.png')
        print(f"    Saving as PNG first at DPI: {target_dpi} (figure size: {fig_width:.1f}\" x {fig_height:.1f}\")")
        fig.savefig(png_path, dpi=target_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   format='png', pad_inches=0.1)
        
        # Try to convert PNG to PDF using PIL if available
        if HAS_PIL:
            try:
                # Disable PIL's decompression bomb check for large images
                Image.MAX_IMAGE_PIXELS = None
                png_img = Image.open(png_path)
                # Convert to RGB if needed
                if png_img.mode != 'RGB':
                    png_img = png_img.convert('RGB')
                # Save as PDF with maximum quality
                png_img.save(output_path, 'PDF', resolution=float(target_dpi), quality=100)
                # Remove temporary PNG if PDF creation succeeded
                png_path.unlink()
                print(f"    Converted PNG to PDF at {target_dpi} DPI")
            except Exception as e:
                print(f"    Warning: Could not convert PNG to PDF: {e}")
                print(f"    Keeping PNG file: {png_path}")
                # Fall back to matplotlib PDF
                fig.savefig(output_path, dpi=target_dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none',
                           format='pdf', pad_inches=0.1,
                           metadata={'Creator': 'MAPseq Figure Generator'})
        else:
            # Direct PDF save if PIL not available
            fig.savefig(output_path, dpi=target_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       format='pdf', pad_inches=0.1,
                       metadata={'Creator': 'MAPseq Figure Generator'})
    else:
        # For other formats, save directly
        fig.savefig(output_path, dpi=target_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   pad_inches=0.1,
                   metadata={'Creator': 'MAPseq Figure Generator'})
    
    plt.close(fig)
    
    print(f"  Saved: {output_path}")
    return output_path


def find_all_parameterizations(base_dir=None):
    """
    Find all parameterization groups across all age groups.
    
    Returns:
        Dictionary: {param_name: {age: param_dir}}
    """
    if base_dir is None:
        base_dir = OUTPUT_BASE
    
    param_groups = {}
    
    for age_dir in base_dir.iterdir():
        if not age_dir.is_dir() or age_dir.name.startswith('.'):
            continue
        
        age = age_dir.name
        if age not in AGE_GROUPS:
            continue
        
        for param_dir in age_dir.iterdir():
            if param_dir.is_dir() and (param_dir.name.startswith('01.') or 
                                      param_dir.name.startswith('02.') or
                                      param_dir.name.startswith('03.') or
                                      param_dir.name.startswith('04.') or
                                      param_dir.name.startswith('05.')):
                param_name = param_dir.name
                
                if param_name not in param_groups:
                    param_groups[param_name] = {}
                
                param_groups[param_name][age] = param_dir
    
    return param_groups


def main():
    parser = argparse.ArgumentParser(
        description="Generate plot matrices for each parameterization group",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures for all parameterizations
  python generate_figure_from_outputs.py
  
  # Generate figure for specific parameterization
  python generate_figure_from_outputs.py --parameterization 05.HAN_filter_parameters_i300_r10_t10_u5
  
  # Custom output directory
  python generate_figure_from_outputs.py --output_dir /path/to/output
        """
    )
    
    parser.add_argument('--parameterization', type=str, default=None,
                       help='Specific parameterization to process (e.g., "05.HAN_filter_parameters_i300_r10_t10_u5")')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for generated figures (default: figure_generation/generated_figures)')
    
    args = parser.parse_args()
    
    # Set output directory
    global FIGURE_OUTPUT_DIR
    if args.output_dir:
        FIGURE_OUTPUT_DIR = Path(args.output_dir)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all parameterizations
    all_param_groups = find_all_parameterizations()
    
    if args.parameterization:
        if args.parameterization in all_param_groups:
            param_groups_to_process = {args.parameterization: all_param_groups[args.parameterization]}
        else:
            print(f"Error: Parameterization '{args.parameterization}' not found.")
            print(f"Available parameterizations: {list(all_param_groups.keys())}")
            return
    else:
        param_groups_to_process = all_param_groups
    
    if not param_groups_to_process:
        print("No parameterizations found.")
        return
    
    print(f"Found {len(param_groups_to_process)} parameterization(s) to process")
    print("=" * 80)
    
    # Process each parameterization
    for param_name, age_param_dirs in param_groups_to_process.items():
        print(f"\nProcessing: {param_name}")
        print(f"  Ages found: {sorted(age_param_dirs.keys())}")
        
        # Save as TIFF for maximum quality (lossless format)
        output_path = FIGURE_OUTPUT_DIR / f"{param_name}_plot_matrix.tiff"
        create_plot_matrix(param_name, age_param_dirs, output_path)
    
    print("\n" + "=" * 80)
    print(f"All figures saved to: {FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
