# Computational pipeline for Culturomics by Automated Microbiome Imaging and Isolation (CAMII)

Future plans include:
- Incorporation of hyperspectral images for diversity-optimized picking (ongoing),
- Image-to-taxonomy models (ongoing),
- End-to-end segmentation and classification.

## Installation

The program runs on Python 3.10+ and can be installed using pip:

```bash
pip install camiira
```

Or install from source:

```bash
git clone https://github.com/whatever60/CAMII_dev.git
cd CAMII_dev
pip install -e .
```

### Dependencies

The package includes the following main dependencies:
- NumPy, SciPy, pandas, Polars, scikit-learn, scikit-image, scikit-bio, biom-format, Matplotlib, seaborn
- opencv-python, Pillow
- python-tsp
- PyYAML, tqdm, rich
- torch, networkx

All dependencies are automatically installed when installing the package.

## Command Line Interface

After installation, the following CLI commands are available:

- `detect_colonies.py` - Colony detection from images
- `calc_calib_params.py` - Calculate calibration parameters  
- `correct_coords.py` - Coordinate correction for picking
- `data_transform.py` - Data transformation utilities
- `draw_interaction.py` - Draw interaction networks
- `isolate_interaction.py` - Isolate interaction inference
- `select_colonies.py` - Colony selection pipeline

## Pipeline description

### Step 0: Prepare your data

Prepare input data. Images and metadata are necessary, and default calibration parameters and picking coordinate correction parameters are provided in the `test_data` directory. Users can define their own calibration parameters and picking coordinate correction parameters, but the default ones are a good starting point.

#### Images

In the current setup, the camera in our robot system takes images of rectangular culture plates under two light conditions:
- Red light from bottom and
- White light along the upper edge of the plate. Images are output in `.bmp` format. Put these image pairs in one directory.

Make sure that:
- File names end with `.bmp`.
- The substring before the first `_` is plate name/barcode. Consequently, plate barcode cannot contain `_`.
- There are two and only two images for each plate, and the image under red light comes before the image under white light when sorted by file name. This is likely already the case since this is the order in which the robot takes pictures.

To proceed, convert the `.bmp` images into `.png` format with

```bash
data_transform.py process_bmp --input_dir <input_dir> --output_dir <output_dir>
```

Images in the output directory will have 4 subdirectories with the following files:
1. `<output_dir>/red_rgb/<plate_barcode>.png`: the picture taken with red light.
2. `<output_dir>/red_grayscale/<plate_barcode>.png`: the picture taken with red light in grayscale.
3. `<output_dir>/white_rgb/<plate_barcode>.png`: the picture taken with white light.
4. `<output_dir>/white_grayscale/<plate_barcode>.png`: the picture taken with white light in grayscale.

You can visually inspect these pictures to assess data quality, but by default the following detection pipeline only uses the grayscale images under red lights.

#### Plate metadata

Prepare a csv file with these columns for each plate (order does not matter and additional columns are allowed though not used):

- `barcode`: Plate barcode.
- `pick_group`: "Picking group" that the plate belongs to. During colony selection steps (see below), optimized picking is performed once for each group.
- `num_picks_group`: Number of colonies to pick from each group.
- `num_picks_plate`: The maximum number of colonies selected from individual plates. This is a constraint for picking optimization and useful when we want to make balance picking across plates. -1 indicates no limit.

#### Config file

A `.yaml` file specifying arguments for the pipeline.

#### Calibration parameter

Based on a reference panel of CAMII pictures, we calculate calibration parameters to account for non-uniform illumination and other artifacts. In the current implementation, we simply do this by dividing the average value of each pixel by the average over the entire image. Input images in the subsequent steps are then divided by these calibration parameters, so that pixels that typically have extreme values are brought closer to the mean, i.e., background is removed.

```bash
calc_calib_params.py --input_dir <input_dir_with_reference_bmp_pairs> --output_dir <output_dir> --config_path <config_file>
```

You can find in this repo pre-computed calibration parameters at `./test_data/parameters/calib_parameter.npz`.

#### Picking coordinate correction parameter:

The robot might not pick where we want, due to lens distortion and other systematic error. We can fit a linear model to correct for this. In current implementation the corrected picking coordinates (`x'` and `y'`) are fitted by:
- `x' = x - (ax2 * x^2 + ax1 * x + axy * y + bx)`
- `y' = y - (ay2 * y^2 + ay1 * y + ayx * x + by)`

where `x` and `y` on the right of the equal sign are the picking coordinates output by the last step, `ax2, ax1, axy, bx, ay2, ay1, ayx, by` are fitted parameters.

Fitting such a model would require actually experimenting with the robot, but a `.json` file with pre-computed model parameters is provided in `./test_data/parameters/correction_params.json`.

### Step 1: Colony detection

Microbial colonies are detected by the canonical pipeline of image processing: background subtraction, thresholding, contour detection, and contour filtering.

```bash
detect_colonies.py \
    --input_path <input_dir_with_the_4_subdirectories> \
    --output_dir <output_dir> \
    --calib_param_path <calibration_parameter_npz> \
    --config_path <config_file>
```

When input path is a `.png` image, colony detection is performed for this single image, and in the output directory, these outputs will be generated:
- `<barcode>_annot.json`, colony segmentation in coco format.
- `<barcode>_gs_red_contour.jpg`, segmentation contours overlaid on the grayscale image under red light.
- `<barcode>_rgb_white_contour.jpg`, segmentation contours overlaid on the RGB image under white light.
- `<barcode>_metadata.csv`, metadata for each contour (i.e., putative colony).

When input path is a directory, colony detection is performed for all `.png` images in the directory, and the same list of output files will be generated for each image in the directory.

### Step 2: Colony selection (initial stage)

A subset of all detected colonies will be selected for picking, under constraint set in the plate metadata. We start by selecting `num_picks_group` colonies (in the metadata) from each group using farthest point algorithm. This algorithm randomly chooses `num_picks_group` colonies and iteratively refines this set until convergence by replacing a colony in the current set by a colony that is farthest away from the current set. When doing replacement, the number of colonies selected for each plate is recorded to not exceed the `num_picks_plate` (in the metadata) limit for each plate.

```bash
select_colonies.py init \
    --image_dir <directory_with_png_images> \
    --input_dir <directory_with_segmentations> \
    --output_dir <output_dir> \
    --metadata_path <path_to_metadata> \
    --config_path <path_to_config>
```

In the output directory, these outputs will be generated for each plate:
- `<barcode>_annot_init.json`, colony segmentation (after initial selection) in coco format.
- `<barcode>_metadata_init.csv`, metadata for each selected contour (i.e., putative colony).
- `<barcode>_gray_contour_init.jpg`, segmentation contours (after initial selection) overlaid on the red light image in grayscale.

### Step 3: Manual inspection

This step serves multiple purposes:
1. Remove false positive colonies detected.
2. Add colonies that were not detected by the algorithm.
3. Manually exclude certain colonies and include others.

However, this can be very laborious and hard to replicate and could be skipped by simply copying the output from the last step to new files by: 

```bash
for file in <output_dir_from_last_step>/*_annot_init.json; do
    dir=$(dirname "$file")
    base_filename=$(basename "$file")
    newbase="${base_filename/_annot_init.json/_annot_init_post.json}"
    newfile="$dir/$newbase"
    cp "$file" "$newfile"
done
```

i.e., adding a `_post` to the filename.

In this step you need to remove unwanted colonies selected in the first step yourself. I suggest quick online tools like [makesense.ai](https://www.makesense.ai/) or [Darwin V7 Lab](https://darwin.v7labs.com/) since you can import and export coco annotations.

After manual tweaking, output segmentation in coco format into the same output directory and name it as `<barcode>_annot_init_post.json`.

### Step 4: Colony selection (post stage)

After a few colonies are labeled as bad colonies, the constraints set in the metadata are no longer satisfied. In this step we run a simpler farthest point algorithm to make up for the lost colonies.

```bash
select_colonies.py post \
    --image_dir <directory_with_png_images> \
    --input_dir <input_dir> \
    --metadata_path <path_to_metadata> \
    --start_from init
```

In the output directory, these outputs will be generated for each plate:
- `<barcode>_annot_final.json`, colony segmentation (after post selection) in coco format.
- `<barcode>_metadata_final.csv`, metadata for each selected contour (i.e., putative colony).
- `<barcode>_gray_contour_final.jpg`, segmentation contours (after post selection) overlaid on the red light image in grayscale.

### Step 5: Go back and forth

If you want you can go back to graphical user interface again to exclude bad colonies. If you do this, store modified segmentation annotation as `<barcode>_annot_final_post.json` in the output directory and run the post selection step again (but make sure to specify `--start_from final` and the output from the last step will be overwritten).

### Step 6: Finalize colony selection

After you are satisfied with the colony selection on each plate, finalize the selection by generating a few visualizations and run Travelling Salesman Problem (TSP) to find the optimal pick order that minimizes robot movement.

```bash
select_colonies.py final \
    -p <directory_with_png_images> \
    -i <input_dir_with_results_from_last_step> \
    -o <output_dir> \
    -m <path_to_metadata> \
    -t [heuristic|exact]
```

In the output directory, these outputs will be generated for each plate:
- `<barcode>_gray_contour.jpg`, segmentation contours overlaid on the red light image in grayscale.
- `<barcode>_metadata.csv`, metadata for each selected contour (i.e., putative colony).
- `<barcode>_picking.json`, picking coordinates of selected colonies in CSV format, first column is x coordinate and second column is y coordinate.
- `<barcode>_rgb_red_contour.jpg`, segmentation contours overlaid on the red light image.
- `<barcode>_rgb_white_contour.jpg`, segmentation contours overlaid on the white light image.

Just note that exact TSP optimization might take forever if you have hundreds of colonies in a plate.

### Step 7 (final step, well done): Coordinate correction

```bash
correct_coords.py \
    -i <input_dir_with_picking_json_from_last_step> \
    -p <correction_parameter_json>
```

In the directory, we do correction for coordinates in all `*_picking.json` files and store corrected coordinates as `<barcode>_Coordinates.csv`, named like this because of the requirement by the colony picking robot.
