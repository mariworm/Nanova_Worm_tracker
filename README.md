Nanova worm tracker is a Python-based program similar to wrMTrck (http://www.phage.dk/plugins/wrmtrck.html).

Currently cannot handle colliding worms! Please only analyze videos where worms stay clear from each other.

Currently it accepts input videos in AVI format and calibration images in TIF format.
You will need a calibration image where you have an object with known dimensions and videos with with worms.
All the videos you want to analyze should be in a separate folder.

Calibration tab: upload an image, draw a line across known distance (you can zoom with mouse wheel), enter the actual size in mm and press calculate, this will set the scale.

Tune parameters tab: upload one of the videos from the batch you are going to analyze, choose a preview (you can switch between previews),
                      check background processing box if you want it to be done, tune threshold and min/max area until worms are selected correctly, 
                      min/max circularity can help filter round dots and dust/hair, so they are not included in the analysis
                      You can run analysis for the uploaded video if needed
                      These parameters will be applied in the batch analysis (next tab)
                      
Batch processing: select the folder with videos and run the analysis, the results will be written in "Nanova_tracker_results.txt" in the same folder.
                      


Tips on performing good quality analysis:

Consistent FPS rate

Consistent uniform lighting, as little gradient and flickering as possible

Plain background without dark inclusions

(When worms collide with dark spots, calculated max speed tend to become significantly higher than it really is)

Only compare worms recorded on the same day, on the same plate/surface, with the same lighting and magnification.


## Version 5.X

### 1. Image Processing & Segmentation

* **Preprocessing Automation:** While `wrmtrck` requires users to manually preprocess videos (like applying background subtraction and thresholding in ImageJ), `Nanova-worm-tracker-5.4.py` automates this natively. It uses an optimized, downscaled rolling-ball background subtraction and direct binary thresholding to isolate dark worms on light backgrounds.


* **Morphological Filtering:** To distinguish worms from background debris, the tracker filters detected contours by Area and a mathematical Circularity formula ($4\pi \times Area / Perimeter^2$). This allows the program to automatically reject perfectly round dust particles and long, thin hairs.


* **Sub-pixel Precision:** Both `Nanova-worm-tracker-5.4.py` and `wrmtrck` calculate the worm's center of mass using image moments (`M["m10"] / M["m00"]`) as floating-point numbers. This sub-pixel precision prevents the spatial quantization (pixel jitter) that occurs if coordinates are rigidly rounded to whole integers.



### 2. Frame-to-Frame Tracking

* **Distance-Based Linking:** `Nanova-worm-tracker-5.4.py` uses a distance matrix and greedy nearest-neighbor assignment to link worm centroids across frames. Similar to `wrmtrck`, the search radius (`max_jump_mm`) is dynamic and scaled by a user-calibrated `px_per_mm` factor, ensuring the linkage threshold is physically consistent regardless of video resolution.


* **Gap Handling:** `Nanova-worm-tracker-5.4.py` allows a track to remain active even if the worm goes undetected for up to 10 consecutive frames.


* **Ambiguity Flagging:** A critical feature of `wrmtrck` is flagging tracks when a frame-to-frame assignment is ambiguous (e.g., more than one candidate in the search radius). `Nanova-worm-tracker-5.4.py` replicates this by flagging steps as ambiguous (`ambig = True`) whenever multiple detections fall within the maximum distance of a track, or multiple tracks fall within the radius of a single detection.



### 3. Trajectory Smoothing

* **Coordinate Smoothing:** To eliminate digital noise and prevent thresholding flicker from inflating path length, both trackers apply a moving average directly to the X and Y coordinates *before* calculating any speeds or distances.


* **Tapered Window:** `Nanova-worm-tracker-5.4.py` implements a wrmtrck-style tapered moving average of up to 5 points. It dynamically shrinks the averaging window near the beginning and end of a track so that every single input frame yields an output coordinate without dropping data at the track edges.



### 4. Kinematics & Output Metrics

* **Speed Calculations:** Step distances are calculated using the smoothed coordinates. The program then calculates path length, average speed, and instantaneous speeds by multiplying the step distances by the video's frames-per-second (FPS).


* **Vmax Protection:** Because overlapping tracking fields can cause "ID swaps" that artificially spike speed, `Nanova-worm-tracker-5.4.py` intentionally masks out any steps flagged as ambiguous when calculating the maximum speed (Vmax), just like `wrmtrck`. These ambiguous steps are still counted toward the total path length.


* **Normalized Speeds:** `wrmtrck` utilizes a metric called BLPS (Body Lengths Per Second) which divides the speed by an approximate worm length calculated as `perimeter / 2`. `Nanova-worm-tracker-5.4.py` uses this exact formula for BLPS, and also introduces an Adjusted Max Speed metric, which normalizes the Vmax by that same approximate body length.


## Version 6.X

The core mathematical and tracking logic changes in "Nanova-worm-tracker-6.3.py" focus on assignment accuracy, ambiguity filtering, and noise reduction:

* **Predicted-Position Gating:** The tracker now calculates a conservative predicted position by adding half of the last recorded displacement to the current position, gating matches more accurately for persistently moving worms.


* **Hybrid Matching Cost:** The cost matrix now includes an area penalty (`D_pred + (lambda_area * area_diff_ratio)`) that weights size differences scaled by half the maximum jump distance, reducing ID swaps when differently sized worms cross paths.


* **Merge Detection Flagging:** Detections are automatically flagged as ambiguous if their current area deviates by more than 40% from the track's recent median area, preventing touching worms from corrupting the max speed calculations.


* **Noise Floor Filtering:** The kinematics calculation now applies a noise floor threshold (`noise_floor_mm_s`) to filter out stationary centroid jitter, zeroing out speeds below this limit before extracting the maximum and 95th-percentile peak speeds.


* **Ambiguity Mask Dilation:** The script dilates the ambiguity mask to cover the entire 5-point smoothing window (a radius of 2 frames), preventing contaminated neighboring steps from leaking into valid speed metrics.
