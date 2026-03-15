import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
# Change this to the exact name of your colorbar/scale file
LEGEND_FILE_NAME = "label.png" 
GRID_COLS = 4
TEXT_SPACE = 120  # Pixels reserved at the top for the title
FONT_SIZE = 90
# ---------------------

def process_maps():
    root_search = Path(__file__).resolve().parent.parent
    image_dir = root_search / "test" / "output" / "individual_defects"
    legend_path = image_dir / LEGEND_FILE_NAME # Assuming it's in the same folder

    if not image_dir.exists():
        print(f"❌ Folder not found at {image_dir}")
        return

    # Find map files, excluding the legend file itself
    files = [f for f in image_dir.glob("map_*.png") if f.name != LEGEND_FILE_NAME]
    files.sort()
    
    if not files:
        print(f"⚠️ No map files found in {image_dir}")
        return

    processed_tiles = []
    # Crop: (Left, Top, Right, Bottom)
    crop_box = (80, 140, 1850, 1690) 

    print(f"--- Processing {len(files)} defects ---")
    
    # Try to load a BOLD font
    try:
        # Standard Windows path for Bold Arial
        font = ImageFont.truetype("arialbd.ttf", FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        except:
            font = ImageFont.load_default()

    for img_path in files:
        defect_name = img_path.stem.split('_')[1].upper()
        
        with Image.open(img_path) as img:
            cropped = img.crop(crop_box)
            cw, ch = cropped.size
            
            # Create a new "Tile" (Image + White space on top)
            tile = Image.new('RGB', (cw, ch + TEXT_SPACE), (255, 255, 255))
            tile.paste(cropped, (0, TEXT_SPACE))
            
            # Draw the BOLD heading
            draw = ImageDraw.Draw(tile)
            # Center the text
            text_bbox = draw.textbbox((0, 0), defect_name, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text(((cw - text_w) // 2, 20), defect_name, fill="black", font=font)
            
            processed_tiles.append(tile)
            print(f"   ✅ Prepared: {defect_name}")

    # Calculate Grid Dimensions
    num = len(processed_tiles)
    rows = (num + GRID_COLS - 1) // GRID_COLS
    tw, th = processed_tiles[0].size
    grid_w = tw * GRID_COLS
    grid_h = th * rows

    # Load Legend
    legend_img = None
    legend_w = 0
    if legend_path.exists():
        legend_img = Image.open(legend_path)
        # Scale legend to match the height of the grid if necessary
        # Or keep its original size if it's already a vertical bar
        legend_w = legend_img.width
        print(f"✅ Found legend file: {LEGEND_FILE_NAME}")
    else:
        print(f"⚠️ Legend file {LEGEND_FILE_NAME} not found. Skipping right-side addition.")

    # Create Final Canvas
    final_canvas = Image.new('RGB', (grid_w + legend_w, grid_h), (255, 255, 255))

    # Paste the Grid
    for i, tile in enumerate(processed_tiles):
        x = (i % GRID_COLS) * tw
        y = (i // GRID_COLS) * th
        final_canvas.paste(tile, (x, y))

    # Paste the Legend on the right
    if legend_img:
        # Center the legend vertically relative to the grid
        ly = (grid_h - legend_img.height) // 2
        final_canvas.paste(legend_img, (grid_w, max(0, ly)))

    # Save to Root Output
    output_dir = root_search / "test" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_report_figure.png"
    final_canvas.save(output_path)
    
    print(f"\n🚀 SUCCESS!")
    print(f"Final figure saved: {output_path}")

if __name__ == "__main__":
    process_maps()