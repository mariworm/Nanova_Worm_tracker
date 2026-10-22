#    Nanova worm tracker
#    Copyright (C) 2026  Maria Ivanova, mariyaiv92@duck.com
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import os
import scipy.spatial.distance as dist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageTk
import threading
import math

# --- CORE TRACKING CLASS ---
class WormTracker:
    def __init__(self, max_distance_px=50, max_gap=10):
        self.tracks = {}
        self.next_id = 1
        self.max_dist = max_distance_px
        self.max_gap = max_gap

    def _new_track(self, d, frame_idx):
        self.tracks[self.next_id] = [{'x': d[0], 'y': d[1], 'area': d[2], 'perim': d[3],
                                      'frame': frame_idx, 'ambig': False}]
        self.next_id += 1

    def update(self, detections, frame_idx):
        if not detections:
            return
        active = [(tid, frame_idx - t[-1]['frame']) for tid, t in self.tracks.items()
                  if 0 < frame_idx - t[-1]['frame'] <= self.max_gap]
        if not active:
            for d in detections:
                self._new_track(d, frame_idx)
            return

        ids = [tid for tid, _ in active]
        gaps = np.array([g for _, g in active])
        
        # 1. Prediction and Area Extraction
        pred_pos, raw_pos, track_areas = [], [], []
        for t in ids:
            trk = self.tracks[t]
            raw_pos.append((trk[-1]['x'], trk[-1]['y']))
            
            # Recent median area (last 5 frames) to establish baseline size
            recent_areas = [d['area'] for d in trk[-5:]]
            track_areas.append(np.median(recent_areas))
            
            # Conservative predicted position (half of last displacement)
            if len(trk) >= 2:
                dx = trk[-1]['x'] - trk[-2]['x']
                dy = trk[-1]['y'] - trk[-2]['y']
                pred_pos.append((trk[-1]['x'] + 0.5 * dx, trk[-1]['y'] + 0.5 * dy))
            else:
                pred_pos.append((trk[-1]['x'], trk[-1]['y']))

        pred_pos = np.array(pred_pos)
        raw_pos = np.array(raw_pos)
        track_areas = np.array(track_areas)[:, None]
        
        det_xy = np.array([(d[0], d[1]) for d in detections])
        det_areas = np.array([d[2] for d in detections])[None, :]
        
        # 2. Hybrid Cost Matrix (Predicted Distance + Area Penalty)
        D_pred = dist.cdist(pred_pos, det_xy)
        area_diff_ratio = np.abs(track_areas - det_areas) / track_areas
        
        # Weight area differences scaled by max_dist (e.g., up to 50% max_dist penalty)
        lambda_area = self.max_dist * 0.5 
        cost_matrix = D_pred + (lambda_area * area_diff_ratio)

        # 3. Spatial Ambiguity (Still based on raw overlap, independent of prediction)
        D_raw = dist.cdist(raw_pos, det_xy)
        ambig_r = np.sum(D_raw <= self.max_dist, axis=1) > 1
        ambig_c = np.sum(D_raw <= self.max_dist, axis=0) > 1

        free_rows = np.ones(len(ids), dtype=bool)
        free_cols = np.ones(len(detections), dtype=bool)
        BIG = 1e9

        for gap in np.unique(gaps):
            rows = np.where((gaps == gap) & free_rows)[0]
            cols = np.where(free_cols)[0]
            if len(rows) == 0 or len(cols) == 0: continue
            
            sub_cost = cost_matrix[np.ix_(rows, cols)]
            sub_dist = D_pred[np.ix_(rows, cols)]
            
            # Gate explicitly by predicted distance, but solve using the hybrid cost
            cost = np.where(sub_dist <= self.max_dist * gap, sub_cost, BIG)
            r_idx, c_idx = linear_sum_assignment(cost)
            
            for ri, ci in zip(r_idx, c_idx):
                if cost[ri, ci] >= BIG: continue
                r, c = rows[ri], cols[ci]
                d = detections[c]
                
                # 4. Merge Detection Flagging (Area deviates > 40%)
                area_deviates = area_diff_ratio[r, c] > 0.4
                is_ambig = bool(ambig_r[r] or ambig_c[c] or gap > 1 or area_deviates)
                
                self.tracks[ids[r]].append({
                    'x': d[0], 'y': d[1], 'area': d[2], 'perim': d[3], 
                    'frame': frame_idx, 'ambig': is_ambig})
                
                free_rows[r] = False
                free_cols[c] = False

        for c in np.where(free_cols)[0]:
            self._new_track(detections[c], frame_idx)


# --- IMAGE PROCESSING HELPER FUNCTION ---
def process_frame(gray_frame, thresh_val, min_a, max_a, min_c, max_c, do_bg_sub, bg_radius):
    # 1. Blur to remove video compression artifacts
    img_blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # 2. ImageJ-Style Rolling Ball Background Subtraction (OPTIMIZED)
    if do_bg_sub and bg_radius > 0:
        # Downscale image for a massive speedup
        scale = 0.25
        small_img = cv2.resize(img_blur, (0, 0), fx=scale, fy=scale)
        
        # Scale the kernel appropriately
        small_k_size = max(3, int((bg_radius * 2 + 1) * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_k_size, small_k_size))
        
        # Calculate background on the tiny image
        small_bg = cv2.morphologyEx(small_img, cv2.MORPH_CLOSE, kernel)
        
        # Upscale background back to original video size
        background = cv2.resize(small_bg, (img_blur.shape[1], img_blur.shape[0]))
        
        # Subtract original from background (Worms become white, background becomes black)
        diff = cv2.subtract(background, img_blur)
        
        # Invert it so background is white and worms are dark
        bg_sub_img = cv2.bitwise_not(diff)
    else:
        bg_sub_img = img_blur.copy()
        
    # 3. Direct threshold for dark objects on a light background.
    _, thresh = cv2.threshold(bg_sub_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
    # 4. Morphological Cleanup (glues broken worm parts together, removes tiny specks)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) 
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_detections, accepted_cnts, rejected_cnts = [], [], []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        
        if perim == 0: continue
        circularity = 4 * np.pi * area / (perim * perim)
        
        if min_a <= area <= max_a and min_c <= circularity <= max_c:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                valid_detections.append((cX, cY, area, perim))
                accepted_cnts.append(cnt)
        else:
            rejected_cnts.append(cnt)
            
    # Return the processed image alongside other data for the preview window
    return valid_detections, accepted_cnts, rejected_cnts, thresh, bg_sub_img

def smooth_positions(arr, max_radius=2):
    """wrmtrck-style tapered moving average: raw at the ends, growing to a
    5-point average in the interior. Every input frame produces an output
    frame — nothing is dropped."""
    n = len(arr)
    out = np.empty(n)
    for i in range(n):
        r = min(i, n - 1 - i, max_radius)
        out[i] = arr[i-r:i+r+1].mean()
    return out


def compute_track_kinematics(data, px_per_mm, fps, noise_floor_mm_s):
    """Enhanced kinematics calculation with frame-gap interpolation, 
    smoothing-aware ambiguity dilation, and robust percentile speed metrics."""
    frames_tracked = len(data)

    # 1. Extract raw coordinates, frame indices, and areas
    frames = np.array([d['frame'] for d in data])
    xs_mm = np.array([d['x'] for d in data]) / px_per_mm
    ys_mm = np.array([d['y'] for d in data]) / px_per_mm
    areas_mm2 = np.array([d['area'] for d in data]) / (px_per_mm ** 2)

    # 2. Linear interpolation across skipped/missing frames
    full_frames = np.arange(frames[0], frames[-1] + 1)
    if len(full_frames) > len(frames):
        xs_interp = np.interp(full_frames, frames, xs_mm)
        ys_interp = np.interp(full_frames, frames, ys_mm)
        
        # Propagate ambiguity flags to interpolated frames
        ambig_raw = np.array([d.get('ambig', False) for d in data])
        ambig_interp = np.interp(full_frames, frames, ambig_raw.astype(float)) > 0
    else:
        xs_interp = xs_mm
        ys_interp = ys_mm
        ambig_interp = np.array([d.get('ambig', False) for d in data])

    # 3. 5-point moving average smoothing
    xs_smooth = smooth_positions(xs_interp, max_radius=2)
    ys_smooth = smooth_positions(ys_interp, max_radius=2)

    # 4. Step distances and instantaneous speeds
    step_dists = np.sqrt(np.diff(xs_smooth)**2 + np.diff(ys_smooth)**2)
    path_length = np.sum(step_dists)
    speeds = step_dists * fps

    # 5. Dilate ambiguity mask to cover the 5-point smoothing window (radius = 2)
    window_radius = 2
    dilated_ambig = np.convolve(np.pad(ambig_interp.astype(int), window_radius),
                            np.ones(2 * window_radius + 1), mode='valid') > 0
    step_ambig_mask = dilated_ambig[:-1] | dilated_ambig[1:]

    valid_speeds = speeds[~step_ambig_mask] if len(speeds) > 0 else np.array([])
    
    # 6. Peak speed metrics with Noise Floor Filtering
    if len(valid_speeds) > 0:
        # Zero out steps that fall beneath the experimental noise floor
        valid_speeds = np.where(valid_speeds < noise_floor_mm_s, 0, valid_speeds)
        max_speed = np.max(valid_speeds)
        p95_speed = np.percentile(valid_speeds, 95)
    else:
        max_speed, p95_speed = np.nan, np.nan

    active_time_sec = (full_frames[-1] - full_frames[0]) / fps if fps > 0 else 0
    avg_speed = path_length / active_time_sec if active_time_sec > 0 else 0

    return {
        'frames_tracked': frames_tracked,
        'xs_mm': xs_mm, 'ys_mm': ys_mm, 'areas_mm2': areas_mm2,
        'path_length': path_length, 
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'p95_speed': p95_speed,
        'active_time_sec': active_time_sec,
    }

# --- MAIN GUI APPLICATION ---
class WormTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nanova Worm Tracker v6")
        self.root.geometry("1200x850")
        
        self.px_per_mm = 1.0
        self.params = {
            'thresh': 100, 'min_a': 100, 'max_a': 1000, 'min_c': 0.05, 'max_c': 0.50, 
            'do_bg_sub': True, 'bg_radius': 50, 'max_jump_mm': 0.3,
            'noise_floor_mm_s': 0.08, 'max_gap_s': 0.4, 'fps': 25.0, 'min_track': 30
        }
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        self.tab_calib = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_calib, text="1. Calibrate Scale")
        self.setup_calib_tab()
        
        self.tab_tune = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tune, text="2. Tune Parameters")
        self.setup_tune_tab()
        
        self.tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_batch, text="3. Batch Processing")
        self.setup_batch_tab()

    
    # --- TAB 1: CALIBRATION ---
    def setup_calib_tab(self):
        control_frame = ttk.Frame(self.tab_calib)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load Calibration Image", command=self.load_calib_image).pack(side="left", padx=5)
        ttk.Label(control_frame, text="(Mouse Wheel = Zoom)").pack(side="left", padx=5)
        
        # New Inline Calibration Controls
        ttk.Label(control_frame, text="Line =").pack(side="left", padx=(20, 2))
        self.ent_ref_mm = ttk.Entry(control_frame, width=5)
        self.ent_ref_mm.insert(0, "1.0")
        self.ent_ref_mm.pack(side="left")
        ttk.Label(control_frame, text="mm").pack(side="left", padx=2)
        
        ttk.Button(control_frame, text="Calculate Scale", command=self.calc_scale_from_line).pack(side="left", padx=10)
        
        ttk.Label(control_frame, text="Scale (px/mm):").pack(side="left", padx=(10, 2))
        self.ent_scale = ttk.Entry(control_frame, width=8, font=('Arial', 10, 'bold'))
        self.ent_scale.insert(0, "1.0")
        self.ent_scale.pack(side="left")
        
        # If the user types manually into the scale box, update the backend variable
        self.ent_scale.bind('<KeyRelease>', self.update_manual_scale)
        
        canvas_frame = ttk.Frame(self.tab_calib)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical")
        self.v_scroll.pack(side="right", fill="y")
        
        self.calib_canvas = tk.Canvas(canvas_frame, bg="gray", cursor="cross", 
                                      xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.calib_canvas.pack(side="left", fill="both", expand=True)
        
        self.h_scroll.config(command=self.calib_canvas.xview)
        self.v_scroll.config(command=self.calib_canvas.yview)
        
        self.calib_canvas.bind("<ButtonPress-1>", self.on_draw_start)
        self.calib_canvas.bind("<B1-Motion>", self.on_draw_drag)
        self.calib_canvas.bind("<ButtonRelease-1>", self.on_draw_release)
        
        self.calib_canvas.bind("<MouseWheel>", self.on_zoom)
        self.calib_canvas.bind("<Button-4>", self.on_zoom)
        self.calib_canvas.bind("<Button-5>", self.on_zoom)
        
        self.draw_line = None
        self.calib_img_rgb = None
        self.calib_zoom = 1.0
        self.last_line_px = 0.0

    def load_calib_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.tif *.tiff *.jpg *.png")])
        if not filepath: return
        img = cv2.imread(filepath)
        self.calib_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.calib_orig_h, self.calib_orig_w = self.calib_img_rgb.shape[:2]
        
        max_width = 1000
        if self.calib_orig_w > max_width:
            self.calib_zoom = max_width / self.calib_orig_w
        else:
            self.calib_zoom = 1.0
            
        self.redraw_calib_image()

    def on_zoom(self, event):
        if self.calib_img_rgb is None: return
        
        # 1. Get the current canvas coordinates of the mouse pointer
        x = self.calib_canvas.canvasx(event.x)
        y = self.calib_canvas.canvasy(event.y)
        
        # 2. Determine zoom scale factor
        scale_factor = 1.1
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            self.calib_zoom *= scale_factor
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            self.calib_zoom /= scale_factor
            scale_factor = 1.0 / scale_factor
        else:
            return
            
        # 3. Redraw the image at the new size (this updates the scrollregion automatically)
        self.redraw_calib_image()
        
        # 4. Calculate where that original pixel moved to on the new resized canvas
        new_canvas_x = x * scale_factor
        new_canvas_y = y * scale_factor
        
        # 5. Calculate the new top-left corner of the viewable window
        new_view_x = new_canvas_x - event.x
        new_view_y = new_canvas_y - event.y
        
        # 6. Get the new total width and height of the canvas
        bbox = self.calib_canvas.bbox("all")
        if not bbox: return
        total_w = bbox[2] - bbox[0]
        total_h = bbox[3] - bbox[1]
        
        # 7. Move the scrollbars to keep the cursor centered exactly where it was
        if total_w > 0 and total_h > 0:
            self.calib_canvas.xview_moveto(new_view_x / total_w)
            self.calib_canvas.yview_moveto(new_view_y / total_h)

    def redraw_calib_image(self):
        if self.draw_line: self.calib_canvas.delete(self.draw_line)
        new_w = int(self.calib_orig_w * self.calib_zoom)
        new_h = int(self.calib_orig_h * self.calib_zoom)
        resized = cv2.resize(self.calib_img_rgb, (new_w, new_h))
        
        self.calib_photo = ImageTk.PhotoImage(image=Image.fromarray(resized))
        self.calib_canvas.create_image(0, 0, image=self.calib_photo, anchor="nw")
        self.calib_canvas.config(scrollregion=self.calib_canvas.bbox("all"))

    def on_draw_start(self, event):
        self.start_x = self.calib_canvas.canvasx(event.x)
        self.start_y = self.calib_canvas.canvasy(event.y)
        if self.draw_line:
            self.calib_canvas.delete(self.draw_line)
        self.draw_line = self.calib_canvas.create_line(self.start_x, self.start_y, self.start_x, self.start_y, fill="lime", width=3)

    def on_draw_drag(self, event):
        cur_x = self.calib_canvas.canvasx(event.x)
        cur_y = self.calib_canvas.canvasy(event.y)
        self.calib_canvas.coords(self.draw_line, self.start_x, self.start_y, cur_x, cur_y)

    def on_draw_release(self, event):
        cur_x = self.calib_canvas.canvasx(event.x)
        cur_y = self.calib_canvas.canvasy(event.y)
        
        px_dist_zoomed = math.hypot(cur_x - self.start_x, cur_y - self.start_y)
        # Save the pixel length quietly in the background without popping up a window
        self.last_line_px = px_dist_zoomed / self.calib_zoom

    def calc_scale_from_line(self):
        if not hasattr(self, 'last_line_px') or self.last_line_px <= 0:
            messagebox.showwarning("Warning", "Please draw a line on the image first.")
            return
            
        try:
            mm = float(self.ent_ref_mm.get())
            if mm <= 0: raise ValueError
            
            # Calculate and set the scale
            self.px_per_mm = self.last_line_px / mm
            
            # Update the entry box to show the newly calculated scale
            self.ent_scale.delete(0, tk.END)
            self.ent_scale.insert(0, f"{self.px_per_mm:.2f}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for Line length.")

    def update_manual_scale(self, event=None):
    # Allows typing a number into the scale box and updating the scale instantly
        try:
            val = float(self.ent_scale.get())
            if val > 0:
                self.px_per_mm = val
        except ValueError:
            pass # Ignore temporary invalid typing (e.g., empty string while backspacing)

    # --- TAB 2: TUNE PARAMETERS ---
    def create_labeled_slider(self, parent, label_text, from_val, to_val, default_val, is_float=False):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        lbl_name = ttk.Label(frame, text=label_text, width=18)
        lbl_name.pack(side="left")
        
        # Format the initial value string
        initial_str = f"{default_val:.2f}" if is_float else str(int(default_val))
        val_str = tk.StringVar(value=initial_str)
        
        # Replace the Label with an Entry widget for manual input
        ent_val = ttk.Entry(frame, textvariable=val_str, width=6, font=('Arial', 9, 'bold'))
        ent_val.pack(side="right")
        
        # 1. When the slider is dragged, update the text box
        def on_slider_move(e):
            val = float(scale.get())
            if not is_float: val = int(val)
            val_str.set(f"{val:.2f}" if is_float else str(val))
            self.update_preview()
            
        scale = ttk.Scale(frame, from_=from_val, to=to_val, value=default_val, command=on_slider_move)
        scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # 2. When text is typed manually, update the slider
        def on_manual_entry(event=None):
            try:
                val = float(val_str.get())
                if not is_float: val = int(val)
                # Keep it within the slider's limits
                val = max(from_val, min(val, to_val)) 
                
                # Update scale (this automatically triggers on_slider_move and updates preview)
                scale.set(val) 
                
                # Re-format the text box cleanly
                val_str.set(f"{val:.2f}" if is_float else str(val))
            except ValueError:
                pass # Ignore invalid typing (like letters) without crashing
                
        # Trigger the manual update when the user presses Enter or clicks away from the text box
        ent_val.bind('<Return>', on_manual_entry)
        ent_val.bind('<FocusOut>', on_manual_entry)
        
        return scale

    def setup_tune_tab(self):
        # --- LEFT PANEL (SCROLLABLE CONTAINER) ---
        # 1. Create a fixed-width container for the canvas and scrollbar
        left_container = ttk.Frame(self.tab_tune, width=380)
        left_container.pack(side="left", fill="y", padx=10, pady=10)
        left_container.pack_propagate(False) # Prevent it from collapsing
        
        # 2. Set up the Canvas and Scrollbar
        self.tune_canvas = tk.Canvas(left_container, highlightthickness=0)
        tune_scroll = ttk.Scrollbar(left_container, orient="vertical", command=self.tune_canvas.yview)
        
        # 3. Create the actual panel that will hold the sliders INSIDE the canvas
        left_panel = ttk.Frame(self.tune_canvas)
        self.left_window = self.tune_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        # 4. Bind configuration events so the scrollbar knows how tall the inner frame gets
        left_panel.bind("<Configure>", lambda e: self.tune_canvas.configure(scrollregion=self.tune_canvas.bbox("all")))
        self.tune_canvas.bind("<Configure>", lambda e: self.tune_canvas.itemconfig(self.left_window, width=e.width))
        self.tune_canvas.configure(yscrollcommand=tune_scroll.set)
        
        self.tune_canvas.pack(side="left", fill="both", expand=True)
        tune_scroll.pack(side="right", fill="y")
        
        # 5. Enable mouse-wheel scrolling only when hovering over the left panel
        def _on_mousewheel(e):
            self.tune_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            
        left_container.bind("<Enter>", lambda e: left_container.bind_all("<MouseWheel>", _on_mousewheel))
        left_container.bind("<Leave>", lambda e: left_container.unbind_all("<MouseWheel>"))

        # --- POPULATE THE LEFT PANEL ---
        ttk.Button(left_panel, text="Load Test Video (.avi)", command=self.load_test_video).pack(fill="x", pady=5)
        self.tune_cap = None
        self.tune_video_path = None
        
        self.view_var = tk.StringVar(value="overlay")
        ttk.Label(left_panel, text="Display View:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        ttk.Radiobutton(left_panel, text="Pre-processed + Contours", variable=self.view_var, value="bg_sub_contours", command=self.update_preview).pack(anchor="w")
        ttk.Radiobutton(left_panel, text="Black & White Mask", variable=self.view_var, value="mask", command=self.update_preview).pack(anchor="w")
        
        self.slider_frame = self.create_labeled_slider(left_panel, "Video Frame", 0, 100, 0)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)

        self.do_bg_var = tk.BooleanVar(value=self.params['do_bg_sub'])
        ttk.Checkbutton(left_panel, text="Apply Rolling Ball BG Subtract", variable=self.do_bg_var, command=self.update_preview).pack(anchor="w", pady=5)
        self.slider_bg_rad = self.create_labeled_slider(left_panel, "Rolling Ball Radius", 1, 200, self.params['bg_radius'])
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)

        # Tracking Parameters
        self.slider_thresh = self.create_labeled_slider(left_panel, "Threshold Value", 0, 255, self.params['thresh'])
        self.slider_min_a = self.create_labeled_slider(left_panel, "Min Area (px)", 10, 2000, self.params['min_a'])
        self.slider_max_a = self.create_labeled_slider(left_panel, "Max Area (px)", 100, 5000, self.params['max_a'])
        self.slider_min_c = self.create_labeled_slider(left_panel, "Min Circularity", 0.0, 1.0, self.params['min_c'], is_float=True)
        self.slider_max_c = self.create_labeled_slider(left_panel, "Max Circularity", 0.0, 1.0, self.params['max_c'], is_float=True)
        self.slider_max_jump = self.create_labeled_slider(left_panel, "Max Jump (mm/frame)", 0.1, 2.0, self.params['max_jump_mm'], is_float=True)
        self.slider_noise_floor = self.create_labeled_slider(left_panel, "Vmax Noise (mm/s)", 0.0, 1.0, self.params['noise_floor_mm_s'], is_float=True)
        self.slider_max_gap = self.create_labeled_slider(left_panel, "Max Gap (s)", 0.0, 2.0, self.params['max_gap_s'], is_float=True)
        
        # Manual Entry Parameters
        frame_fps = ttk.Frame(left_panel)
        frame_fps.pack(fill="x", pady=5)
        ttk.Label(frame_fps, text="Video FPS", width=18).pack(side="left")
        self.ent_fps = ttk.Entry(frame_fps, width=8, font=('Arial', 9, 'bold'))
        self.ent_fps.insert(0, str(self.params['fps']))
        self.ent_fps.pack(side="right")
        self.ent_fps.bind('<KeyRelease>', self.update_preview)

        frame_min_track = ttk.Frame(left_panel)
        frame_min_track.pack(fill="x", pady=5)
        ttk.Label(frame_min_track, text="Min Track (frames)", width=18).pack(side="left")
        self.ent_min_track = ttk.Entry(frame_min_track, width=8, font=('Arial', 9, 'bold'))
        self.ent_min_track.insert(0, str(self.params['min_track']))
        self.ent_min_track.pack(side="right")
        self.ent_min_track.bind('<KeyRelease>', self.update_preview)
        
        self.lbl_stats = ttk.Label(left_panel, text="Worms Found: 0", foreground="blue", font=('Arial', 10, 'bold'))
        self.lbl_stats.pack(pady=10)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)
        
        # Keep Frame Range sliders on the left
        ttk.Label(left_panel, text="Test Frame Range:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(5, 0))
        self.slider_start_frame = self.create_labeled_slider(left_panel, "Start Frame", 0, 100, 0)
        self.slider_end_frame = self.create_labeled_slider(left_panel, "End Frame", 0, 100, 100)
        
        
        # --- RIGHT PANEL (SCROLLABLE CONTAINER) ---
        right_container = ttk.Frame(self.tab_tune)
        right_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.right_canvas = tk.Canvas(right_container, highlightthickness=0)
        right_scroll = ttk.Scrollbar(right_container, orient="vertical", command=self.right_canvas.yview)
        
        right_panel = ttk.Frame(self.right_canvas)
        self.right_window = self.right_canvas.create_window((0, 0), window=right_panel, anchor="nw")
        
        right_panel.bind("<Configure>", lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))
        self.right_canvas.bind("<Configure>", lambda e: self.right_canvas.itemconfig(self.right_window, width=e.width))
        self.right_canvas.configure(yscrollcommand=right_scroll.set)
        
        self.right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.pack(side="right", fill="y")
        
        def _on_mousewheel_right(e):
            self.right_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            
        right_container.bind("<Enter>", lambda e: right_container.bind_all("<MouseWheel>", _on_mousewheel_right))
        right_container.bind("<Leave>", lambda e: right_container.unbind_all("<MouseWheel>"))

        # --- POPULATE THE RIGHT PANEL ---
        self.lbl_video_title = ttk.Label(right_panel, text="No video loaded", font=('Arial', 10, 'bold'), anchor="center", wraplength=700)
        self.lbl_video_title.pack(side="top", fill="x", pady=(0, 5))
        
        self.lbl_preview = ttk.Label(right_panel, text="Load a video to preview...", anchor="n")
        self.lbl_preview.pack(side="top", fill="both", expand=True, anchor="n")
        
        # Test Output Block packed below the video sequentially using side="top"
        output_frame = ttk.Frame(right_panel)
        output_frame.pack(side="top", fill="x", pady=10)
        
        self.btn_test_run = ttk.Button(output_frame, text="Run Analysis on This Video", command=self.start_test_analysis)
        self.btn_test_run.pack(fill="x", pady=5)
        
        self.test_output = tk.Text(output_frame, height=8, width=40, font=('Consolas', 8))
        self.test_output.pack(fill="x", pady=5)

    def load_test_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("AVI files", "*.avi")])
        if not filepath: return
        self.tune_video_path = filepath
        
        # Display the full file path above the preview
        self.lbl_video_title.config(text=f"Video: {filepath}")
        
        if self.tune_cap: self.tune_cap.release()
        self.tune_cap = cv2.VideoCapture(filepath)
        total_frames = int(self.tune_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        max_f = max(1, total_frames - 1)
        self.slider_frame.config(to=max_f)
        self.slider_frame.set(total_frames // 2)
        
        if hasattr(self, 'slider_start_frame'):
            self.slider_start_frame.config(to=max_f)
            self.slider_start_frame.set(0)
            
        if hasattr(self, 'slider_end_frame'):
            self.slider_end_frame.config(to=max_f)
            self.slider_end_frame.set(max_f)
            
        self.update_preview()

    def update_preview(self, event=None):
        if not self.tune_cap: return
        
        t_val = int(self.slider_thresh.get())
        min_a = int(self.slider_min_a.get())
        max_a = int(self.slider_max_a.get())
        min_c = float(self.slider_min_c.get())
        max_c = float(self.slider_max_c.get())
        max_jump = float(self.slider_max_jump.get())
        do_bg = self.do_bg_var.get()
        bg_rad = int(self.slider_bg_rad.get())
        max_gap = float(self.slider_max_gap.get())
        noise_floor = float(self.slider_noise_floor.get())
        
        # Parse manual FPS input with fallback
        try:
            fps_val = float(self.ent_fps.get())
            if fps_val <= 0: raise ValueError
        except ValueError:
            fps_val = 25.0

        # Parse manual Min Track input with fallback
        MIN_TRACK_FLOOR = 5   # 5-point smoothing is meaningless below this

        try:
            min_tr_val = int(self.ent_min_track.get())
            if min_tr_val <= 0: raise ValueError
        except ValueError:
            min_tr_val = 30
        min_tr_val = max(MIN_TRACK_FLOOR, min_tr_val)
        
        self.params = {
            'thresh': t_val, 'min_a': min_a, 'max_a': max_a, 
            'min_c': min_c, 'max_c': max_c, 'do_bg_sub': do_bg, 
            'bg_radius': bg_rad, 'max_jump_mm': max_jump,
            'max_gap_s': max_gap, 'fps': fps_val, 'min_track': min_tr_val,
            'noise_floor_mm_s': noise_floor
        }

        f_idx = int(self.slider_frame.get())
        self.tune_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = self.tune_cap.read()
        if not ret: return
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Now unpacks 5 variables
        valid, acc_cnt, rej_cnt, mask, bg_sub_img = process_frame(gray, t_val, min_a, max_a, min_c, max_c, do_bg, bg_rad)
        
        self.lbl_stats.config(text=f"Worms Found (Green): {len(acc_cnt)}\nDropped (Red): {len(rej_cnt)}")
        
        # Display rendering logic
        if self.view_var.get() == "mask":
            disp_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif self.view_var.get() == "bg_sub":
            disp_frame = cv2.cvtColor(bg_sub_img, cv2.COLOR_GRAY2BGR)
        elif self.view_var.get() == "bg_sub_contours":
            disp_frame = cv2.cvtColor(bg_sub_img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(disp_frame, rej_cnt, -1, (0, 0, 255), 1)
            cv2.drawContours(disp_frame, acc_cnt, -1, (0, 255, 0), 2)
        else:
            disp_frame = frame.copy()
            cv2.drawContours(disp_frame, rej_cnt, -1, (0, 0, 255), 1)
            cv2.drawContours(disp_frame, acc_cnt, -1, (0, 255, 0), 2)
        
        max_h = 700
        if disp_frame.shape[0] > max_h:
            scale = max_h / disp_frame.shape[0]
            disp_frame = cv2.resize(disp_frame, (int(disp_frame.shape[1]*scale), max_h))
            
        disp_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
        self.preview_img = ImageTk.PhotoImage(image=Image.fromarray(disp_rgb))
        self.lbl_preview.config(image=self.preview_img, text="")

    # --- SINGLE VIDEO TEST ANALYSIS ---
    def start_test_analysis(self):
        if not self.tune_video_path: return
        self.btn_test_run.config(state="disabled")
        self.test_output.delete('1.0', tk.END)
        self.test_output.insert(tk.END, "Analyzing video...\nThis may take a moment.")
        threading.Thread(target=self.run_single_test, daemon=True).start()

    def run_single_test(self):
        cap = cv2.VideoCapture(self.tune_video_path)
        fps = float(self.params.get('fps', 25.0))
        max_dist_px = self.params.get('max_jump_mm', 0.3) * self.px_per_mm
        max_gap = max(1, round(self.params.get('max_gap_s', 0.4) * fps))
        tracker = WormTracker(max_distance_px=max_dist_px, max_gap=max_gap)
        
        # Pull Min Track settings from self.params
        min_track = int(self.params.get('min_track', 30))
        
        start_f = int(self.slider_start_frame.get())
        end_f = int(self.slider_end_frame.get())
        
        if start_f > end_f:
            start_f, end_f = end_f, start_f
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        f_idx = start_f
        
        while f_idx <= end_f:
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            valid_det, _, _, _, _ = process_frame(
                gray, self.params['thresh'], self.params['min_a'], 
                self.params['max_a'], self.params['min_c'], self.params['max_c'],
                self.params['do_bg_sub'], self.params['bg_radius']
            )
            tracker.update(valid_det, f_idx)
            f_idx += 1
            
        cap.release()
        
        all_results = []
        
        for tid, data in tracker.tracks.items():
            if len(data) < min_track: continue
            data = sorted(data, key=lambda k: k['frame'])
            
            m = compute_track_kinematics(
                data, 
                self.px_per_mm, 
                fps, 
                noise_floor_mm_s=self.params.get('noise_floor_mm_s', 0.0)
            )

            perims_mm = np.array([d['perim'] for d in data]) / self.px_per_mm
            euclid_dist = np.sqrt((m['xs_mm'][-1] - m['xs_mm'][0])**2 + (m['ys_mm'][-1] - m['ys_mm'][0])**2)
            avg_area = np.mean(m['areas_mm2'])
            avg_perim = np.mean(perims_mm)
            length_approx = avg_perim / 2.0
            blps = m['avg_speed'] / length_approx if length_approx > 0 else 0
            total_time_sec = round(m['active_time_sec'], 2)

            all_results.append({
                'Track_ID': tid, 
                'Frames': m['frames_tracked'], 
                'Time_sec': total_time_sec,
                'Distance_mm': round(euclid_dist, 3), 
                'Path_Length_mm': round(m['path_length'], 3),
                'Avg_Speed_mm_s': round(m['avg_speed'], 4), 
                'Max_Speed_mm_s': round(m['max_speed'], 4),
                'P95_Speed_mm_s': round(m['p95_speed'], 4),
                'Avg_Area_mm2': round(avg_area, 4), 
                'sd_Area_mm2': round(np.std(m['areas_mm2']), 4),
                'Avg_Perim_mm': round(avg_perim, 4), 
                'sd_Perim_mm': round(np.std(perims_mm), 4),
                'BLPS': round(blps, 4)
            })

        if all_results:
            df = pd.DataFrame(all_results)
            report = f"--- TEST RESULTS ---\nFound {len(all_results)} valid tracks (>{min_track} frames).\n\n"
            report += df.to_csv(sep='\t', index=False)
        else:
            report = "--- TEST RESULTS ---\nNo valid tracks found."
            
        self.root.after(0, lambda: self.test_output.delete('1.0', tk.END))
        self.root.after(0, lambda: self.test_output.insert(tk.END, report))
        self.root.after(0, lambda: self.btn_test_run.config(state="normal"))

    # --- TAB 3: BATCH PROCESSING ---
    def setup_batch_tab(self):
        frame = ttk.Frame(self.tab_batch)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.lbl_folder = ttk.Label(frame, text="No folder selected", font=('Arial', 10, 'italic'))
        self.lbl_folder.pack(pady=5)
        ttk.Button(frame, text="1. Select Video Folder", command=self.select_batch_folder).pack()
        
        input_frame = ttk.Frame(frame)
        input_frame.pack(pady=20)
        
        self.btn_run = ttk.Button(frame, text="2. RUN BATCH ANALYSIS", command=self.start_batch_thread)
        self.btn_run.pack(pady=10)
        
        self.progress = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)
        
        self.log_txt = tk.Text(frame, height=10, state='disabled')
        self.log_txt.pack(fill="x", pady=10)
        self.batch_dir = ""

    def log(self, msg):
        self.log_txt.config(state='normal')
        self.log_txt.insert(tk.END, msg + "\n")
        self.log_txt.see(tk.END)
        self.log_txt.config(state='disabled')
        self.root.update_idletasks()

    def select_batch_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.batch_dir = folder
            self.lbl_folder.config(text=f"Selected: {self.batch_dir}")
            
    def start_batch_thread(self):
        if not self.batch_dir:
            messagebox.showerror("Error", "Please select a folder first.")
            return
        self.btn_run.config(state="disabled")
        self.log("Starting batch process... Please wait.")
        threading.Thread(target=self.run_batch, daemon=True).start()

    def run_batch(self):
        fps = float(self.params.get('fps', 25.0))
        min_track = int(self.params.get('min_track', 30))

        files = [f for f in os.listdir(self.batch_dir) if f.endswith('.avi')]
        if not files:
            self.root.after(0, lambda: self.log("No .avi files found in directory."))
            self.root.after(0, lambda: self.btn_run.config(state="normal"))
            return

        all_results = []
        total_files = len(files)
        
        for i, file in enumerate(files):
            self.root.after(0, lambda f=file: self.log(f"Processing: {f}"))
            vid_path = os.path.join(self.batch_dir, file)
            
            cap = cv2.VideoCapture(vid_path)
            max_dist_px = self.params.get('max_jump_mm', 0.3) * self.px_per_mm
            max_gap = max(1, round(self.params.get('max_gap_s', 0.4) * fps))
            tracker = WormTracker(max_distance_px=max_dist_px, max_gap=max_gap)
            f_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Dummy variable for the preview image
                valid_det, _, _, _, _ = process_frame(gray, self.params['thresh'], self.params['min_a'], 
                                                   self.params['max_a'], self.params['min_c'], self.params['max_c'],
                                                   self.params['do_bg_sub'], self.params['bg_radius'])
                tracker.update(valid_det, f_idx)
                f_idx += 1
                
            cap.release()
            
            for tid, data in tracker.tracks.items():
                if len(data) < min_track: continue
                data = sorted(data, key=lambda k: k['frame'])
                
                m = compute_track_kinematics(
                    data, 
                    self.px_per_mm, 
                    fps, 
                    noise_floor_mm_s=self.params.get('noise_floor_mm_s', 0.0)
                )

                perims_mm = np.array([d['perim'] for d in data]) / self.px_per_mm
                euclid_dist = np.sqrt((m['xs_mm'][-1] - m['xs_mm'][0])**2 + (m['ys_mm'][-1] - m['ys_mm'][0])**2)
                avg_area = np.mean(m['areas_mm2'])
                avg_perim = np.mean(perims_mm)
                length_approx = avg_perim / 2.0
                blps = m['avg_speed'] / length_approx if length_approx > 0 else 0
                total_time_sec = round(m['active_time_sec'], 2)

                all_results.append({
                    'File': file, 
                    'Track_ID': tid, 
                    'Frames': m['frames_tracked'], 
                    'Time_sec': total_time_sec,
                    'Distance_mm': round(euclid_dist, 3), 
                    'Path_Length_mm': round(m['path_length'], 3),
                    'Avg_Speed_mm_s': round(m['avg_speed'], 4), 
                    'Max_Speed_mm_s': round(m['max_speed'], 4),
                    'P95_Speed_mm_s': round(m['p95_speed'], 4),
                    'Avg_Area_mm2': round(avg_area, 4), 
                    'sd_Area_mm2': round(np.std(m['areas_mm2']), 4),
                    'Avg_Perim_mm': round(avg_perim, 4), 
                    'sd_Perim_mm': round(np.std(perims_mm), 4),
                    'BLPS': round(blps, 4)
                })
            
            self.root.after(0, lambda val=int(((i+1)/total_files)*100): self.progress.configure(value=val))

        if all_results:
            df = pd.DataFrame(all_results)
            save_path = os.path.join(self.batch_dir, 'Nanova_tracker_results.txt')
            df.to_csv(save_path, sep='\t', index=False)
            self.root.after(0, lambda: self.log(f"\nDONE! Results saved to:\n{save_path}"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Analysis complete!\nSaved to: {save_path}"))
        else:
            self.root.after(0, lambda: self.log("\nFinished, but no valid tracks found."))
            
        self.root.after(0, lambda: self.btn_run.config(state="normal"))

if __name__ == "__main__":
    root = tk.Tk()
    app = WormTrackerApp(root)
    root.mainloop()
