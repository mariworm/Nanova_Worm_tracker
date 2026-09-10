Nanova worm tracker is a Python-based program, ideological successor of wrMTrck (http://www.phage.dk/plugins/wrmtrck.html).
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
