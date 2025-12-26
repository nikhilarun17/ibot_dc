import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Sketch functions

def pencil_sketch(image, kernel_size=(21, 21),verify=True):
    if verify:
        image = cv2.imread(image).astype(np.uint8)
    img_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img_invert = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_invert, kernel_size, 0)
    img_blur_invert = 255 - img_blur
    img_sketch = np.clip((img_gray / img_blur_invert) * 256, 0, 255).astype(np.uint8)
    return img_sketch

def colour_sketch(image, kernel_size=(21, 21), desaturation=0, verify=True):
    if verify:
        image = cv2.imread(image)
    hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_inv = 255 - v
    v_blur = cv2.GaussianBlur(v_inv, kernel_size, 0)
    sketch_v = cv2.divide(v, 255 - v_blur, scale=256).astype(np.uint8)

    s = np.clip(s * (1 - desaturation), 0, 255).astype(np.uint8)
    hsv_sketch = cv2.merge([h, s, sketch_v])
    return cv2.cvtColor(hsv_sketch, cv2.COLOR_HSV2BGR)

def detect_circles(image, dp=1, min_dist=25, param1=50, param2=50, min_radius=10, max_radius=400, verify=True):
    #Detect circles and color code them based on size
    if verify:
        image = cv2.imread(image).astype(np.uint8)
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp, 
                               min_dist, 
                               param1=param1, 
                               param2=param2, 
                               minRadius=min_radius, 
                               maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
    
    result = image.copy()
    if circles is not None:
        # Color code circles by size
        radii = circles[0, :, 2]
        mean_radius = np.mean(radii)
        std_radius = np.std(radii)
        
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            
            # Classify circle size
            if r < mean_radius - std_radius:
                color = (255, 0, 0)  # Blue for small
            elif r > mean_radius + std_radius:
                color = (0, 0, 255)  # Red for large
            else:
                color = (0, 255, 0)  # Green for medium
            
            # Draw circle outline
            cv2.circle(result, (x, y), r, color, 2)
            # Draw center
            cv2.circle(result, (x, y), 2, (0, 255, 255), 3)
    return result

# Tinker GUI

class TinkerGUI:
    def __init__(self,root):
        self.root = root
        root.title("CV Tinker GUI")
        root.geometry("3200x2000")

        self.finalpath=None
        
        self.frame=tk.Frame(root,padx=100,pady=50)
        self.frame.grid(row=0,column=0,sticky="n")

        # Input path and browse button
        tk.Label(self.frame, text="Input Image/video:", font=("Arial", 10)).grid(row=0, column=0, sticky='w')
        self.path_var = tk.StringVar(master=self.root)
        tk.Entry(self.frame, textvariable=self.path_var, width=40).grid(row=1, column=0, columnspan=3)
        tk.Button(self.frame, text="Browse...", command=self.open_imvid).grid(row=1, column=3, padx=4)
        
        # Method selection (pencil/colour/circles)
        tk.Label(self.frame, text="Method:", font=("Arial", 10)).grid(row=2, column=0, sticky='w', pady=(8, 0))
        self.method_var = tk.StringVar(value='pencil')
        tk.Radiobutton(self.frame, text="Pencil (grayscale)", variable=self.method_var, value='pencil', font=("Arial", 10)).grid(row=3, column=0, columnspan=4, sticky='w')
        tk.Radiobutton(self.frame, text="Colour (HSV)", variable=self.method_var,value='hsv', font=("Arial", 10)).grid(row=4, column=0, columnspan=4, sticky='w')
        tk.Radiobutton(self.frame, text="Circle Detector (Hough)", variable=self.method_var,value='circles', font=("Arial", 10)).grid(row=5, column=0, columnspan=4, sticky='w')


        # Kernel size input
        tk.Label(self.frame, text="Gaussian kernel (odd ints):", font=("Arial", 10)).grid(row=6, column=0, sticky='w', pady=(8, 0))
        tk.Label(self.frame, text="(Width,Height):", font=("Arial", 10)).grid(row=7, column=0, sticky='e')
        self.kw = tk.Spinbox(self.frame, from_=1, to=101, increment=2, width=5, font=("Arial", 10))
        self.kw.delete(0, 'end')
        self.kw.insert(0, '21')
        self.kw.grid(row=7, column=1, sticky='w')
        if int(self.kw.get())%2==0:
            messagebox.showinfo('Info', 'Kernel size should be odd integers!')

        # Image size input
        tk.Label(self.frame, text="Image-Size:", font=("Arial", 10)).grid(row=8, column=0, sticky='w', pady=(8, 0))
        self.image_size_var = tk.Spinbox(self.frame, from_=100, to=1200, increment=50, width=7, font=("Arial", 10))
        self.image_size_var.delete(0, 'end')
        self.image_size_var.insert(0, '600')
        self.image_size_var.grid(row=8, column=0, sticky='e', padx=200)

        # HSV desaturation slider
        tk.Label(self.frame, text="HSV desaturation (for HSV method):", font=("Arial", 10)).grid(row=10, column=0, sticky='w', pady=(8, 0))
        self.desat = tk.Scale(self.frame, from_=0.0, to=0.8, resolution=0.01, orient='horizontal', length=180, font=("Arial", 10))
        self.desat.set(0.15)
        self.desat.grid(row=9, column=0, columnspan=4)

        # Circle Detector parameters
        tk.Label(self.frame, text="Circle Detection Parameters:", font=("Arial", 10)).grid(row=11, column=0, sticky='w', pady=(4, 2))
        
        tk.Label(self.frame, text="Parameter 1:", font=("Arial", 10)).grid(row=13, column=0, sticky='w', padx=(0, 5))
        self.param1 = tk.Scale(self.frame, from_=10, to=200, orient='horizontal', length=100, highlightthickness=0, font=("Arial", 10))
        self.param1.set(50)
        self.param1.grid(row=13, column=1, columnspan=2, sticky='w', padx=(0, 10))

        tk.Label(self.frame, text="Parameter 2:", font=("Arial", 10)).grid(row=14, column=0, sticky='w', padx=(0, 5))
        self.param2 = tk.Scale(self.frame, from_=10, to=200, orient='horizontal', length=100, highlightthickness=0, font=("Arial", 10))
        self.param2.set(50)
        self.param2.grid(row=14, column=1, columnspan=2, sticky='w', padx=(0, 10))

        tk.Label(self.frame, text="Minimum Radius:", font=("Arial", 10)).grid(row=15, column=0, sticky='w', padx=(0, 5))
        self.min_radius = tk.Scale(self.frame, from_=5, to=100, orient='horizontal', length=100, highlightthickness=0, font=("Arial", 10))
        self.min_radius.set(10)
        self.min_radius.grid(row=15, column=1, columnspan=2, sticky='w', padx=(0, 10))

        tk.Label(self.frame, text="Maximum Radius:", font=("Arial", 10)).grid(row=16, column=0, sticky='w', padx=(0, 5))
        self.max_radius = tk.Scale(self.frame, from_=100, to=500, orient='horizontal', length=100, highlightthickness=0, font=("Arial", 10))
        self.max_radius.set(400)
        self.max_radius.grid(row=16, column=1, columnspan=2, sticky='w', padx=(0, 10))
        
        self.auto_adjust_var = tk.BooleanVar(value=False)

        # Run, Save, Quit buttons
        tk.Button(self.frame, text="Run", width=10, command=self.run_sketch, font=("Arial", 10)).grid(row=18, column=0, pady=(10, 0))
        tk.Button(self.frame, text="Auto Analyze", width=12, command=self.auto_analyse, font=("Arial", 10)).grid(row=18, column=1, pady=(10, 0), padx=(5, 0))
        tk.Button(self.frame, text="Save Result", width=12, command=self.save_result, font=("Arial", 10)).grid(row=18, column=2, pady=(10, 0), padx=(5, 0))
        tk.Button(self.frame, text="Quit", width=8, command=root.quit, font=("Arial", 10)).grid(row=18, column=3, pady=(10, 0))

        # Display frame for original and result media
        self.display_frame = tk.Frame(root, padx=8, pady=8)
        self.display_frame.grid(row=0, column=1)
        self.orig_media = tk.Label(self.display_frame)
        self.orig_media.grid(row=1, column=0)
        self.result_media = tk.Label(self.display_frame)
        self.result_media.grid(row=1, column=1)
        tk.Label(self.display_frame, text="Original", font=("Arial", 10)).grid(row=0, column=0)
        tk.Label(self.display_frame, text="Result", font=("Arial", 10)).grid(row=0, column=1)

        # Status bar (idk y now)
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var, font=("Arial", 9)).grid(row=2, column=0, columnspan=2, sticky='we')

        #Video layout variables
        self.video1_paused=True
        self.video2_paused=True
        self.video1_button=tk.Button(self.display_frame,text="Play Original",command=self.toggle_video1, font=("Arial", 9))
        self.video1_button.grid(row=2,column=0,pady=4)
        self.video2_button=tk.Button(self.display_frame,text="Play Result",command=self.toggle_video2, font=("Arial", 9))
        self.video2_button.grid(row=2,column=1,pady=4)
        self.video1=None
        self.video2=None
        self.sketch_func=None
        self.fps=30
        self.current_frame_no=0
        self.temp_switch=False
        self.video_loop_running=False
        self.frame_count=0  # Counter for slider updates

    def start_video_loop(self):
        # Start the video update loop if not already running (just running update_videos repeatedly creates infinite loops)
        if not self.video_loop_running:
            self.video_loop_running = True
            self.update_videos()
    
    #Video loading functions
    def load_video1(self,path):
        if self.video1:
            self.video1.release()
        self.video1=cv2.VideoCapture(path)
    def load_video2(self,path):
        if self.video2:
            self.video2.release()
        self.video2=cv2.VideoCapture(path)
    def analyze_frame_params(self, frame):
        """Analyze a single frame and return optimal parameters."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Calculate edge density using Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Calculate contrast
        contrast = gray.std()
        
        # Estimate parameters based on image properties
        if contrast < 30:
            param1 = 40
        elif contrast < 60:
            param1 = 50
        else:
            param1 = 70
        
        if edge_density < 0.05:
            param2 = 30
        elif edge_density < 0.15:
            param2 = 50
        else:
            param2 = 70
        
        min_radius = max(5, int(min(height, width) * 0.02))
        max_radius = min(500, int(min(height, width) * 0.4))
        
        return param1, param2, min_radius, max_radius
    
    def update_videos(self):
        if self.video1 and not self.video1_paused:
            ret1, frame1 = self.video1.read()
            self.current_frame_no=int(self.video1.get(cv2.CAP_PROP_POS_FRAMES))
            if ret1:
                self.show_on_label(frame1, self.orig_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
        if self.video2 and not self.video2_paused:
            ret2, frame2 = self.video2.read()
            if ret2:
                if self.sketch_func == 'pencil_sketch':
                    frame2 = pencil_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), verify=False)
                elif self.sketch_func == 'colour_sketch':
                    frame2 = colour_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), desaturation=self.desat.get(),verify=False)
                elif self.sketch_func == 'detect_circles':
                    # Analyze current frame and adjust parameters dynamically if enabled
                    if self.auto_adjust_var.get():
                        param1, param2, min_radius, max_radius = self.analyze_frame_params(frame2)
                        # Update sliders every 10 frames for visibility
                        self.frame_count += 1
                        if self.frame_count % 10 == 0:
                            self.param1.set(param1)
                            self.param2.set(param2)
                            self.min_radius.set(min_radius)
                            self.max_radius.set(max_radius)
                            self.root.update_idletasks()
                    else:
                        param1, param2, min_radius, max_radius = self.param1.get(), self.param2.get(), self.min_radius.get(), self.max_radius.get()
                    frame2 = detect_circles(frame2,
                                           param1=param1,
                                           param2=param2,
                                           min_radius=min_radius,
                                           max_radius=max_radius,
                                           verify=False)
                self.show_on_label(frame2, self.result_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
        
        delay = int(1000 / self.fps)
        self.root.after(delay, self.update_videos)
    def show_on_label(self, img, label, maxsize=(600,600)):
        img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        im_pil = im_pil.resize((maxsize))
        im_tk = ImageTk.PhotoImage(im_pil)
        label.img_tk = im_tk
        label.config(image = im_tk)
    def toggle_video1(self):
        self.video1_paused = not self.video1_paused
        if self.video1_paused:
            self.video1_button.config(text="Play Original")
        else:
            self.video1_button.config(text="Pause Original")
    def toggle_video2(self):
        self.video2_paused = not self.video2_paused
        if self.video2_paused:
            self.video2_button.config(text="Play Result")
        else:
            self.video2_button.config(text="Pause Result")
    # Open image/video function
    def open_imvid(self):
        self.video1_paused,self.video2_paused=True,True
        self.current_frame_no=0
        self.path = filedialog.askopenfilename(title='Select image or video', filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("Video files", "*.mp4 *.avi *.mkv *.mov *.webm *.mpeg *.mpg"),("All files", "*.*")])
        if not self.path:
            return
        self.image_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.video_ext = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpeg", ".mpg"} 
        self.ext = os.path.splitext(self.path)[1].lower()
        self.path_var.set(os.path.basename(self.path))
        if self.ext in self.image_ext:
            print("Image selected")
            try:
                img = cv2.imread(self.path)
                if img is None:
                    raise ValueError('Failed to read image')
                self.show_on_label(img, self.orig_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
                self.status_var.set(f'Loaded: {os.path.basename(self.path)}')
            except Exception as e:
                messagebox.showerror('Error', f'Could not load image: {e}')
                self.status_var.set('Error loading image')
        elif self.ext in self.video_ext:
            print("Video selected")
            self.video1_paused = True
            self.status_var.set(f'Playing video: {os.path.basename(self.path)}')
            self.root.update_idletasks()
            self.load_video1(self.path)
            self.video1.set(cv2.CAP_PROP_POS_FRAMES,0)
            ret1, frame1 = self.video1.read()
            if ret1:
                self.show_on_label(frame1, self.orig_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
        for widget in [self.result_media]:
            widget.config(image='', text='')
            self.temp_switch=False
    # Run sketch function
    def run_sketch(self):
        if int(self.kw.get())%2==0:
            messagebox.showinfo('Info', 'Kernel size should be odd integers!')
        if self.temp_switch and not(self.video1_paused and self.video2_paused):
            self.video1_paused,self.video2_paused=True,True
            self.video1_button.config(text="Play Original")
            self.video2_button.config(text="Play Result")
        elif self.temp_switch and (self.video1_paused and self.video2_paused):
            self.video1_paused,self.video2_paused=False,False
            self.video1_button.config(text="Pause Original")
            self.video2_button.config(text="Pause Result")
        kernel_width = int(self.kw.get())
        method = self.method_var.get()
        if self.ext in self.image_ext:
            if method == 'pencil':
                result = pencil_sketch(self.path, kernel_size=(kernel_width, kernel_width))
            elif method == 'hsv':
                result = colour_sketch(self.path, kernel_size=(kernel_width, kernel_width), desaturation=self.desat.get())
            elif method == 'circles':
                result = detect_circles(self.path, 
                                       param1=self.param1.get(), 
                                       param2=self.param2.get(),
                                       min_radius=self.min_radius.get(),
                                       max_radius=self.max_radius.get(),
                                       verify=True)
            self.show_on_label(result, self.result_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
            self.finalpath=result
            self.status_var.set('Processing completed')
        elif self.ext in self.video_ext:
            if method == 'pencil':
                self.sketch_func = 'pencil_sketch'
            elif method == 'hsv':
                self.sketch_func = 'colour_sketch'
            elif method == 'circles':
                self.sketch_func = 'detect_circles'
    
            self.video2_paused = self.video1_paused
            self.status_var.set(f'Playing sketched video: {os.path.basename(self.path)}')
            self.root.update_idletasks()
            self.load_video2(self.path)
            self.video2.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame_no)
            self.frame_count = 0  # Reset frame counter
            ret2, frame2 = self.video2.read()
            self.temp_switch=True
            if ret2 and self.video2_paused:
                if self.sketch_func == 'pencil_sketch':
                    frame2 = pencil_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), verify=False)
                elif self.sketch_func == 'colour_sketch':
                    frame2 = colour_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), desaturation=self.desat.get(),verify=False)
                elif self.sketch_func == 'detect_circles':
                    # Analyze current frame and adjust parameters dynamically if enabled
                    if self.auto_adjust_var.get():
                        param1, param2, min_radius, max_radius = self.analyze_frame_params(frame2)
                        # Update sliders to reflect dynamic parameters
                        self.param1.set(param1)
                        self.param2.set(param2)
                        self.min_radius.set(min_radius)
                        self.max_radius.set(max_radius)
                    else:
                        param1, param2, min_radius, max_radius = self.param1.get(), self.param2.get(), self.min_radius.get(), self.max_radius.get()
                    frame2 = detect_circles(frame2, 
                                           param1=param1,
                                           param2=param2,
                                           min_radius=min_radius,
                                           max_radius=max_radius,
                                           verify=False)
                self.show_on_label(frame2, self.result_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
            self.start_video_loop()
    
    def auto_analyse(self):
        """Automatically analyze the image/video and set optimal circle detection parameters."""
        if not self.path:
            messagebox.showwarning('Warning', 'Please load an image or video first!')
            return
        
        try:
            # Helper function to analyze a frame
            def analyze_frame(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                
                # Calculate edge density using Canny edge detection
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (height * width)
                
                # Calculate contrast
                contrast = gray.std()
                
                # Estimate parameters based on image properties
                # Param1: Edge threshold - higher for low contrast images
                if contrast < 30:
                    param1 = 40
                elif contrast < 60:
                    param1 = 50
                else:
                    param1 = 70
                
                # Param2: Center threshold - adjust based on edge density
                if edge_density < 0.05:
                    param2 = 30
                elif edge_density < 0.15:
                    param2 = 50
                else:
                    param2 = 70
                
                # Estimate radius range based on image size
                min_radius = max(5, int(min(height, width) * 0.02))
                max_radius = min(500, int(min(height, width) * 0.4))
                
                return param1, param2, min_radius, max_radius, height, width
            
            # Handle images
            if self.ext in self.image_ext:
                img = cv2.imread(self.path)
                if img is None:
                    raise ValueError('Failed to read image')
                
                param1, param2, min_radius, max_radius, height, width = analyze_frame(img)
                
                # Set the parameters
                self.param1.set(param1)
                self.param2.set(param2)
                self.min_radius.set(min_radius)
                self.max_radius.set(max_radius)
                
                self.status_var.set(f'Auto-analyzed: P1={param1}, P2={param2}, MinR={min_radius}, MaxR={max_radius}')
            
            # Handle videos
            elif self.ext in self.video_ext:
                video = cv2.VideoCapture(self.path)
                ret, frame = video.read()
                if not ret:
                    raise ValueError('Failed to read video')
                
                param1, param2, min_radius, max_radius, height, width = analyze_frame(frame)
                
                # Set the parameters
                self.param1.set(param1)
                self.param2.set(param2)
                self.min_radius.set(min_radius)
                self.max_radius.set(max_radius)
                
                # Set circle detection method and run automatically
                self.method_var.set('circles')
                self.status_var.set(f'Auto-analyzed video: P1={param1}, P2={param2}, MinR={min_radius}, MaxR={max_radius} - Running...')
                self.root.update_idletasks()
                
                # Run the circle detection
                self.run_sketch()
                
                video.release()
        
        except Exception as e:
            messagebox.showerror('Error', f'Could not analyze: {e}')
            self.status_var.set('Error during analysis')
    
    # Save result function for only images since video saving is a pain
    def save_result(self):

        if self.ext in self.image_ext:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")])
            if not save_path:
                return
            try:
                if isinstance(self.finalpath, np.ndarray):
                    cv2.imwrite(save_path, self.finalpath)
                self.status_var.set(f'Saved result to: {os.path.basename(save_path)}')
            except Exception as e:
                messagebox.showerror('Error', f'Could not save image: {e}')
                self.status_var.set('Error saving image')
        if self.ext in self.video_ext:
            messagebox.showinfo('Info', 'Implement it yourself! :)')

if __name__ == "__main__":
    root = tk.Tk()
    app = TinkerGUI(root)
    root.mainloop()