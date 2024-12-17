import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import cv2
import numpy as np

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")  # Dark mode
ctk.set_default_color_theme("dark-blue")

# Dictionary of YOLO Models
YOLO_MODELS = {
    "yolov5n": "yolov5n_new/weights/best.pt",
    "yolov5m": "yolov5m_new/weights/best.pt",
    "yolov8n": "yolov8n_new/weights/best.pt",
    "yolov8m": "yolov8m_new/weights/best.pt"
}

# Main Application Class
class YOLOInferenceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Inference Tool")
        self.geometry("900x600")
        
        # Variables
        self.models = list(YOLO_MODELS.keys())  # Model names for dropdown
        self.selected_model = ctk.StringVar(value=self.models[0])
        self.image_path = None
        self.output_image = None
        self.original_image_size = None
        
        # Layout
        self.create_sidebar()
        self.create_main_canvas()
    
    def create_sidebar(self):
        # Sidebar Frame
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        # Dropdown for Model Selection
        self.model_label = ctk.CTkLabel(self.sidebar, text="Select YOLO Model:", font=("Arial", 14))
        self.model_label.pack(pady=(20, 10))
        
        self.model_dropdown = ctk.CTkOptionMenu(self.sidebar, values=self.models, variable=self.selected_model)
        self.model_dropdown.pack(pady=10, padx=10)
        
        # Upload Image Button
        self.upload_button = ctk.CTkButton(self.sidebar, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10, padx=10)
        
        # Infer Button
        self.infer_button = ctk.CTkButton(self.sidebar, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(pady=10, padx=10)
        
        # Save Image Button
        self.save_button = ctk.CTkButton(self.sidebar, text="Save Processed Image", command=self.save_image)
        self.save_button.pack(pady=10, padx=10)
    
    def create_main_canvas(self):
        # Canvas to Display Image
        self.canvas_frame = ctk.CTkFrame(self, corner_radius=0)
        self.canvas_frame.pack(pady=10, expand=True, fill="both")
        self.canvas = ctk.CTkCanvas(self.canvas_frame, bg="black")
        self.canvas.pack(expand=False)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image(self.image_path)
    
    def display_image(self, image_path):
        image = Image.open(image_path)
        self.original_image_size = image.size  # Store original size
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.configure(width=self.original_image_size[0], height=self.original_image_size[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas_frame.configure(width=self.original_image_size[0], height=self.original_image_size[1])
    
    def run_inference(self):
        if not self.image_path:
            print("Please upload an image first.")
            return
        
        # Load YOLO model
        model_name = self.selected_model.get()
        model_path = YOLO_MODELS[model_name]
        print(f"Running inference with model: {model_name} ({model_path})")
        model = YOLO(model_path)
        
        # Run inference
        results = model(self.image_path, save=False, imgsz=640, verbose=False)
        result_image = results[0].plot()  # Get the result image
        
        # Convert result image (BGR -> RGB)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        self.output_image = Image.fromarray(result_image)
        self.display_processed_image()
    
    def display_processed_image(self):
        # Display image at original resolution
        self.tk_image = ImageTk.PhotoImage(self.output_image)
        self.canvas.configure(width=self.original_image_size[0], height=self.original_image_size[1])
        self.canvas_frame.configure(width=self.original_image_size[0], height=self.original_image_size[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        print("Inference completed and image displayed.")
    
    def save_image(self):
        if self.output_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if save_path:
                self.output_image.save(save_path)
                print(f"Image saved to {save_path}")
        else:
            print("No processed image to save.")

# Run Application
if __name__ == "__main__":
    app = YOLOInferenceApp()
    app.mainloop()