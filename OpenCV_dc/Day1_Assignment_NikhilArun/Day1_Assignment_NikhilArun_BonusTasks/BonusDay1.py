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
    v_blur_inv = 255 - v_blur
    sketch_v = cv2.divide(v, v_blur_inv, scale=256).astype(np.uint8)

    s = np.clip(s * (1 - desaturation), 0, 255).astype(np.uint8)
    hsv_sketch = cv2.merge([h, s, sketch_v])
    return cv2.cvtColor(hsv_sketch, cv2.COLOR_HSV2BGR)

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
        tk.Label(self.frame, text="Input Image/video:").grid(row=0, column=0, sticky='w')
        self.path_var = tk.StringVar(master=self.root)
        tk.Entry(self.frame, textvariable=self.path_var, width=40).grid(row=1, column=0, columnspan=3)
        tk.Button(self.frame, text="Browse...", command=self.open_imvid).grid(row=1, column=3, padx=4)
        
        # Method selection (pencil/colour)
        tk.Label(self.frame, text="Method:").grid(row=2, column=0, sticky='w', pady=(8, 0))
        self.method_var = tk.StringVar(value='pencil')
        tk.Radiobutton(self.frame, text="Pencil (grayscale)", variable=self.method_var, value='pencil').grid(row=3, column=0, columnspan=4, sticky='w')
        tk.Radiobutton(self.frame, text="Colour (HSV)", variable=self.method_var,value='hsv').grid(row=5, column=0, columnspan=4, sticky='w')

        # Kernel size input
        tk.Label(self.frame, text="Gaussian kernel (odd ints):").grid(row=6, column=0, sticky='w', pady=(8, 0))
        tk.Label(self.frame, text="(Width,Height):").grid(row=7, column=0, sticky='e')
        self.kw = tk.Spinbox(self.frame, from_=1, to=101, increment=2, width=5)
        self.kw.delete(0, 'end')
        self.kw.insert(0, '21')
        self.kw.grid(row=7, column=1, sticky='w')
        if int(self.kw.get())%2==0:
            messagebox.showinfo('Info', 'Kernel size should be odd integers!')

        # Image size input
        tk.Label(self.frame, text="Image-Size:").grid(row=8, column=0, sticky='w', pady=(8, 0))
        self.image_size_var = tk.Spinbox(self.frame, from_=100, to=1200, increment=50, width=7)
        self.image_size_var.delete(0, 'end')
        self.image_size_var.insert(0, '600')
        self.image_size_var.grid(row=8, column=0, sticky='e', padx=200)

        # HSV desaturation slider
        tk.Label(self.frame, text="HSV desaturation (for HSV method):").grid(row=10, column=0, sticky='w', pady=(8, 0))
        self.desat = tk.Scale(self.frame, from_=0.0, to=0.8, resolution=0.01, orient='horizontal', length=180)
        self.desat.set(0.15)
        self.desat.grid(row=9, column=0, columnspan=4)

        # Run, Save, Quit buttons
        tk.Button(self.frame, text="Run", width=10, command=self.run_sketch).grid(row=12, column=0, pady=(10, 0))
        tk.Button(self.frame, text="Save Result", width=12, command=self.save_result).grid(row=12, column=1, columnspan=2, pady=(10, 0))
        tk.Button(self.frame, text="Quit", width=8, command=root.quit).grid(row=12, column=3, pady=(10, 0))

        # Display frame for original and result media
        self.display_frame = tk.Frame(root, padx=8, pady=8)
        self.display_frame.grid(row=0, column=1)
        self.orig_media = tk.Label(self.display_frame)
        self.orig_media.grid(row=1, column=0)
        self.result_media = tk.Label(self.display_frame)
        self.result_media.grid(row=1, column=1)
        tk.Label(self.display_frame, text="Original").grid(row=0, column=0)
        tk.Label(self.display_frame, text="Result").grid(row=0, column=1)

        # Status bar (idk y now)
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(root, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky='we')

        #Video layout variables
        self.video1_paused=True
        self.video2_paused=True
        self.video1_button=tk.Button(self.display_frame,text="Play Original",command=self.toggle_video1)
        self.video1_button.grid(row=2,column=0,pady=4)
        self.video2_button=tk.Button(self.display_frame,text="Play Result",command=self.toggle_video2)
        self.video2_button.grid(row=2,column=1,pady=4)
        self.video1=None
        self.video2=None
        self.sketch_func=None
        self.fps=30
        self.current_frame_no=0
        self.temp_switch=False
        self.video_loop_running=False

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
            self.show_on_label(result, self.result_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
            self.finalpath=result
            self.status_var.set('Sketch generated')
        elif self.ext in self.video_ext:
            if method == 'pencil':
                self.sketch_func = 'pencil_sketch'
            elif method == 'hsv':
                self.sketch_func = 'colour_sketch'
    
            self.video2_paused = self.video1_paused
            self.status_var.set(f'Playing sketched video: {os.path.basename(self.path)}')
            self.root.update_idletasks()
            self.load_video2(self.path)
            self.video2.set(cv2.CAP_PROP_POS_FRAMES,self.current_frame_no)
            ret2, frame2 = self.video2.read()
            self.temp_switch=True
            if ret2 and self.video2_paused:
                if self.sketch_func == 'pencil_sketch':
                    frame2 = pencil_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), verify=False)
                elif self.sketch_func == 'colour_sketch':
                    frame2 = colour_sketch(frame2, kernel_size=(int(self.kw.get()), int(self.kw.get())), desaturation=self.desat.get(),verify=False)
                self.show_on_label(frame2, self.result_media, maxsize=(int(self.image_size_var.get()),int(self.image_size_var.get())))
            self.start_video_loop()
    
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
