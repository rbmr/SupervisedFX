from pathlib import Path
import xml.etree.ElementTree as ET
import re

def get_path_bbox(path_element):
    """
    Parses the 'd' attribute of an SVG <path> to find its bounding box.
    This is a simplified parser for Matplotlib's typical path structures.
    """
    d = path_element.get('d', '')
    # Use regex to find all numeric coordinates in the path data
    points = [float(p) for p in re.findall(r'[-]?\d+\.\d*|[-]?\d+', d)]
    
    if not points:
        return 0, 0, 0, 0

    x_coords = points[0::2]
    y_coords = points[1::2]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    return min_x, min_y, max_x, max_y

def move_and_crop_svg(input_svg_path, output_svg_path, padding=10.0):
    """
    Moves the legend in a Matplotlib-generated SVG to the bottom-right corner
    of the main plot area and then crops the SVG canvas to remove extra space.

    Args:
        input_svg_path (str): The path to the original SVG file.
        output_svg_path (str): The path where the modified SVG will be saved.
        padding (float): The padding in pixels for positioning and cropping.
    """
    try:
        ET.register_namespace('', "http://www.w3.org/2000/svg")
        tree = ET.parse(input_svg_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error reading or parsing SVG file: {e}")
        return

    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    # === 1. FIND ELEMENTS (Same as before) ===
    axes_bbox_path = root.find('.//svg:g[@id="axes_1"]/svg:g[@id="patch_2"]/svg:path', namespaces)
    legend_group = root.find('.//svg:g[@id="legend_1"]', namespaces)
    
    if axes_bbox_path is None or legend_group is None:
        print("Could not find required axes or legend elements. Aborting.")
        return
        
    legend_bbox_path = legend_group.find('.//svg:g/svg:path', namespaces)
    if legend_bbox_path is None:
        print("Could not find the legend bounding box path. Aborting.")
        return

    # === 2. MOVE LEGEND (Same as before) ===
    ax_min_x, ax_min_y, ax_max_x, ax_max_y = get_path_bbox(axes_bbox_path)
    leg_min_x, leg_min_y, leg_max_x, leg_max_y = get_path_bbox(legend_bbox_path)
    leg_width = leg_max_x - leg_min_x
    leg_height = leg_max_y - leg_min_y
    
    target_x = ax_max_x - leg_width - padding
    target_y = ax_max_y - leg_height - padding
    translate_x = target_x - leg_min_x
    translate_y = target_y - leg_min_y
    
    legend_group.set('transform', f'translate({translate_x},{translate_y})')
    print("Step 1: Legend moved successfully.")

    # === 3. CROP SVG CANVAS (New functionality) ===
    print("Step 2: Cropping SVG canvas...")
    original_viewbox_str = root.get('viewBox')
    original_width_str = root.get('width')

    if not original_viewbox_str or not original_width_str:
        print("Could not find 'viewBox' or 'width' attributes on root SVG element. Cannot crop.")
        return

    # Extract original dimensions
    vb_min_x, vb_min_y, vb_width, vb_height = map(float, original_viewbox_str.split())
    
    try:
        # Handle units like 'pt', 'px', etc.
        val_match = re.search(r'[\d\.]+', original_width_str)
        unit_match = re.search(r'[a-zA-Z]+', original_width_str)
        original_phys_width_val = float(val_match.group(0)) if val_match else 0
        original_phys_width_unit = unit_match.group(0) if unit_match else ""
    except (AttributeError, IndexError):
        print(f"Could not parse original width: '{original_width_str}'")
        return

    # Calculate the new width based on the axes' right edge
    new_vb_width = ax_max_x + padding
    
    # Calculate the new physical width proportionally
    ratio = new_vb_width / vb_width
    new_phys_width = original_phys_width_val * ratio
    
    print(f"  - Original viewBox width: {vb_width:.2f}")
    print(f"  - New viewBox width: {new_vb_width:.2f}")
    print(f"  - Cropping SVG to {new_phys_width:.2f}{original_phys_width_unit}")

    # Update the root SVG element's width and viewBox
    root.set('viewBox', f'{vb_min_x} {vb_min_y} {new_vb_width} {vb_height}')
    root.set('width', f'{new_phys_width:.2f}{original_phys_width_unit}')
    
    # Also update the figure's background patch to match the new size
    figure_patch = root.find('.//svg:g[@id="figure_1"]/svg:g[@id="patch_1"]/svg:path', namespaces)
    if figure_patch is not None:
        new_d = f"M {vb_min_x} {vb_height} L {new_vb_width} {vb_height} L {new_vb_width} {vb_min_y} L {vb_min_x} {vb_min_y} z"
        figure_patch.set('d', new_d)

    # === 4. SAVE FINAL IMAGE ===
    tree.write(output_svg_path)
    print(f"\nSuccess! Moved and cropped SVG saved to '{output_svg_path}'")


if __name__ == '__main__':
    # --- USAGE ---
    # 1. Place your SVG file in the same directory as this script.
    # 2. Change 'input.svg' to the name of your file.
    # 3. Change 'output_with_moved_legend.svg' to your desired output filename.
    for file in Path('.').glob('*.svg'):
        if file.name.startswith('moved_'):
            continue
        print(f"Processing file: {file.name}")
        move_and_crop_svg(file.name, f"moved_{file.name}")