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
from PIL import Image, ImageTk
import threading
import math

# --- CORE TRACKING CLASS ---
class WormTracker:
    def __init__(self, max_distance_px=50):
        self.tracks = {}
        self.next_id = 1
        self.max_dist = max_distance_px
        
    def update(self, detections, frame_idx):
        if len(self.tracks) == 0:
            for d in detections:
                self.tracks[self.next_id] = [{'x': d[0], 'y': d[1], 'area': d[2], 'perim': d[3], 'frame': frame_idx, 'ambig': False}]
                self.next_id += 1
            return
            
        active_ids, active_pos = [], []
        for tid, data in self.tracks.items():
            if frame_idx - data[-1]['frame'] < 10: 
                active_ids.append(tid)
                active_pos.append((data[-1]['x'], data[-1]['y']))
                
        if not active_ids or not detections:
            for d in detections:
                self.tracks[self.next_id] = [{'x': d[0], 'y': d[1], 'area': d[2], 'perim': d[3], 'frame': frame_idx, 'ambig': False}]
                self.next_id += 1
            return
            
        D = dist.cdist(active_pos, [(d[0], d[1]) for d in detections])
        
        # --- AMBIGUITY CHECK ---
        # A track is ambiguous if >1 detection is within radius. A detection is ambiguous if >1 track is within radius.
        ambig_r = np.sum(D <= self.max_dist, axis=1) > 1
        ambig_c = np.sum(D <= self.max_dist, axis=0) > 1
        
        used_rows, used_cols = set(), set()
        for _ in range(min(D.shape[0], D.shape[1])):
            if D.min() > self.max_dist: break
            r, c = np.unravel_index(D.argmin(), D.shape)
            if r not in used_rows and c not in used_cols:
                is_ambig = bool(ambig_r[r] or ambig_c[c])
                self.tracks[active_ids[r]].append({
                    'x': detections[c][0], 'y': detections[c][1], 
                    'area': detections[c][2], 'perim': detections[c][3], 'frame': frame_idx, 'ambig': is_ambig
                })
                used_rows.add(r)
                used_cols.add(c)
                D[r, :] = np.inf
                D[:, c] = np.inf
                
        for c, d in enumerate(detections):
            if c not in used_cols:
                self.tracks[self.next_id] = [{'x': d[0], 'y': d[1], 'area': d[2], 'perim': d[3], 'frame': frame_idx, 'ambig': False}]
                self.next_id += 1


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


def compute_track_kinematics(data, px_per_mm, fps):
    """Core kinematics shared by the single-video test view and the batch
    pipeline. `data` is a track's per-frame detections, sorted by frame."""
    frames_tracked = len(data)

    xs_mm = np.array([d['x'] for d in data]) / px_per_mm
    ys_mm = np.array([d['y'] for d in data]) / px_per_mm
    areas_mm2 = np.array([d['area'] for d in data]) / (px_per_mm ** 2)

    xs_smooth = smooth_positions(xs_mm)
    ys_smooth = smooth_positions(ys_mm)

    step_dists = np.sqrt(np.diff(xs_smooth) ** 2 + np.diff(ys_smooth) ** 2)
    path_length = np.sum(step_dists)
    speeds = step_dists * fps

    ambig_flags = np.array([d.get('ambig', False) for d in data])
    ambig_mask = ambig_flags[:-1] | ambig_flags[1:] if len(ambig_flags) > 1 else np.array([], dtype=bool)

    valid_speeds = speeds[~ambig_mask] if len(speeds) > 0 else np.array([])
    max_speed = np.max(valid_speeds) if len(valid_speeds) > 0 else 0

    active_time_sec = len(step_dists) / fps if fps > 0 else 0
    avg_speed = path_length / active_time_sec if active_time_sec > 0 else 0

    return {
        'frames_tracked': frames_tracked,
        'xs_mm': xs_mm, 'ys_mm': ys_mm, 'areas_mm2': areas_mm2,
        'path_length': path_length, 'avg_speed': avg_speed,
        'max_speed': max_speed, 'active_time_sec': active_time_sec,
    }

# --- MAIN GUI APPLICATION ---
class WormTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nanova Worm Tracker v5")
        self.root.geometry("1200x850")
        
        self.px_per_mm = 1.0
        self.params = {
            'thresh': 100, 'min_a': 100, 'max_a': 1000, 'min_c': 0.05, 'max_c': 0.50, 
            'do_bg_sub': False, 'bg_radius': 50
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
        # This allows you to just type a number into the scale box and have it update instantly
        try:
            val = float(self.ent_scale.get())
            if val > 0:
                self.px_per_mm = val
        except ValueError:
            pass # Ignore temporary invalid typing (like empty string while deleting)
        
        def save_scale():
            try:
                mm = float(entry.get())
                self.px_per_mm = px_dist_original / mm
                self.lbl_scale.config(text=f"Current Scale: {self.px_per_mm:.2f} px/mm")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number.")
                
        ttk.Button(dialog, text="Save", command=save_scale).pack(pady=10)

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
        left_panel = ttk.Frame(self.tab_tune, width=350)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)
        
        ttk.Button(left_panel, text="Load Test Video (.avi)", command=self.load_test_video).pack(fill="x", pady=5)
        self.tune_cap = None
        self.tune_video_path = None
        
        # Display Option - Now includes Pre-processed Video
        self.view_var = tk.StringVar(value="overlay")
        ttk.Label(left_panel, text="Display View:", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(10,0))
        ttk.Radiobutton(left_panel, text="Pre-processed + Contours", variable=self.view_var, value="bg_sub_contours", command=self.update_preview).pack(anchor="w")
        ttk.Radiobutton(left_panel, text="Black & White Mask", variable=self.view_var, value="mask", command=self.update_preview).pack(anchor="w")
        
        # Video Frame Slider
        self.slider_frame = self.create_labeled_slider(left_panel, "Video Frame", 0, 100, 0)
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)

        # Background Subtraction
        self.do_bg_var = tk.BooleanVar(value=self.params['do_bg_sub'])
        ttk.Checkbutton(left_panel, text="Apply Rolling Ball BG Subtract", variable=self.do_bg_var, command=self.update_preview).pack(anchor="w", pady=5)
        self.slider_bg_rad = self.create_labeled_slider(left_panel, "Rolling Ball Radius", 10, 200, self.params['bg_radius'])
        
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)

        # Main Parameters
        self.slider_thresh = self.create_labeled_slider(left_panel, "Threshold Value", 0, 255, self.params['thresh'])
        self.slider_min_a = self.create_labeled_slider(left_panel, "Min Area (px)", 10, 2000, self.params['min_a'])
        self.slider_max_a = self.create_labeled_slider(left_panel, "Max Area (px)", 100, 5000, self.params['max_a'])
        self.slider_min_c = self.create_labeled_slider(left_panel, "Min Circularity", 0.0, 1.0, self.params['min_c'], is_float=True)
        self.slider_max_c = self.create_labeled_slider(left_panel, "Max Circularity", 0.0, 1.0, self.params['max_c'], is_float=True)
        self.slider_max_jump = self.create_labeled_slider(left_panel, "Max Jump (mm/frame)", 0.1, 5.0, 1.0, is_float=True)
        
        self.lbl_stats = ttk.Label(left_panel, text="Worms Found: 0", foreground="blue", font=('Arial', 10, 'bold'))
        self.lbl_stats.pack(pady=10)
        
        # Test Analysis Functionality
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=10)
        self.btn_test_run = ttk.Button(left_panel, text="Run Analysis on This Video", command=self.start_test_analysis)
        self.btn_test_run.pack(fill="x", pady=5)
        
        self.test_output = tk.Text(left_panel, height=8, width=40, font=('Consolas', 8))
        self.test_output.pack(fill="x", pady=5)
        
        self.lbl_preview = ttk.Label(self.tab_tune, text="Load a video to preview...")
        self.lbl_preview.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    def load_test_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("AVI files", "*.avi")])
        if not filepath: return
        self.tune_video_path = filepath
        
        if self.tune_cap: self.tune_cap.release()
        self.tune_cap = cv2.VideoCapture(filepath)
        total_frames = int(self.tune_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider_frame.config(to=max(1, total_frames-1))
        self.slider_frame.set(total_frames // 2)
        
        self.update_preview()

    def update_preview(self, event=None):
        if not self.tune_cap: return
        
        f_idx = int(self.slider_frame.get())
        t_val = int(self.slider_thresh.get())
        min_a = int(self.slider_min_a.get())
        max_a = int(self.slider_max_a.get())
        min_c = float(self.slider_min_c.get())
        max_c = float(self.slider_max_c.get())
        max_jump = float(self.slider_max_jump.get())
        do_bg = self.do_bg_var.get()
        bg_rad = int(self.slider_bg_rad.get())
        
        self.params = {'thresh': t_val, 'min_a': min_a, 'max_a': max_a, 'min_c': min_c, 'max_c': max_c, 'do_bg_sub': do_bg, 'bg_radius': bg_rad, 'max_jump_mm': max_jump}
        
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
        max_dist_px = self.params.get('max_jump_mm', 1.0) * self.px_per_mm
        tracker = WormTracker(max_distance_px=max_dist_px)
        f_idx = 0
        fps = 25.0 
        
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
        
        all_results = []
        min_track = 30 
        
        for tid, data in tracker.tracks.items():
            if len(data) < min_track: continue
            data = sorted(data, key=lambda k: k['frame'])
            
            m = compute_track_kinematics(data, self.px_per_mm, fps)

            perims_mm = np.array([d['perim'] for d in data]) / self.px_per_mm
            euclid_dist = np.sqrt((m['xs_mm'][-1]-m['xs_mm'][0])**2 + (m['ys_mm'][-1]-m['ys_mm'][0])**2)
            avg_area = np.mean(m['areas_mm2'])
            avg_perim = np.mean(perims_mm)
            length_approx = avg_perim / 2.0
            blps = m['avg_speed'] / length_approx if length_approx > 0 else 0
            adj_max_speed = m['max_speed'] / length_approx if length_approx > 0 else 0
            total_time_sec = m['frames_tracked'] / fps

            all_results.append({
                'Track_ID': tid, 'Frames': m['frames_tracked'], 'Time_sec': round(total_time_sec, 2),
                'Distance_mm': round(euclid_dist, 3), 'Path_Length_mm': round(m['path_length'], 3),
                'Avg_Speed_mm_s': round(m['avg_speed'], 4), 'Max_Speed_mm_s': round(m['max_speed'], 4),
                'Avg_Area_mm2': round(avg_area, 4), 'sd_Area_mm2': round(np.std(m['areas_mm2']), 4),
                'Avg_Perim_mm': round(avg_perim, 4), 'sd_Perim_mm': round(np.std(perims_mm), 4),
                'BLPS': round(blps, 4), 'Adj_Max_Speed': round(adj_max_speed, 4)
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
        
        ttk.Label(input_frame, text="Video FPS:").grid(row=0, column=0, padx=5)
        self.ent_fps = ttk.Entry(input_frame, width=10)
        self.ent_fps.insert(0, "25.0")  
        self.ent_fps.grid(row=0, column=1)
        
        ttk.Label(input_frame, text="Min Track Length (frames):").grid(row=1, column=0, padx=5, pady=10)
        self.ent_min_track = ttk.Entry(input_frame, width=10)
        self.ent_min_track.insert(0, "30")
        self.ent_min_track.grid(row=1, column=1)
        
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
        try:
            fps = float(self.ent_fps.get())
            min_track = int(self.ent_min_track.get())
        except ValueError:
            self.root.after(0, lambda: messagebox.showerror("Error", "Check FPS and Min Track inputs."))
            self.root.after(0, lambda: self.btn_run.config(state="normal"))
            return

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
            max_dist_px = self.params.get('max_jump_mm', 1.0) * self.px_per_mm
            tracker = WormTracker(max_distance_px=max_dist_px)
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
                
                m = compute_track_kinematics(data, self.px_per_mm, fps)

                perims_mm = np.array([d['perim'] for d in data]) / self.px_per_mm
                euclid_dist = np.sqrt((m['xs_mm'][-1]-m['xs_mm'][0])**2 + (m['ys_mm'][-1]-m['ys_mm'][0])**2)
                avg_area = np.mean(m['areas_mm2'])
                avg_perim = np.mean(perims_mm)
                length_approx = avg_perim / 2.0
                blps = m['avg_speed'] / length_approx if length_approx > 0 else 0
                adj_max_speed = m['max_speed'] / length_approx if length_approx > 0 else 0
                total_time_sec = m['frames_tracked'] / fps

                all_results.append({
                    'File': file, 'Track_ID': tid, 'Frames': m['frames_tracked'], 'Time_sec': round(total_time_sec, 2),
                    'Distance_mm': round(euclid_dist, 3), 'Path_Length_mm': round(m['path_length'], 3),
                    'Avg_Speed_mm_s': round(m['avg_speed'], 4), 'Max_Speed_mm_s': round(m['max_speed'], 4),
                    'Avg_Area_mm2': round(avg_area, 4), 'sd_Area_mm2': round(np.std(m['areas_mm2']), 4),
                    'Avg_Perim_mm': round(avg_perim, 4), 'sd_Perim_mm': round(np.std(perims_mm), 4),
                    'BLPS': round(blps, 4), 'Adj_Max_Speed': round(adj_max_speed, 4)
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
