"""
CS 5001
Spring 2025
Final Project – Milestone 2: Gradio Web App & Additional Filters
This file contains the Gradio interface for the image processing application.
Author: 
Ziming "Alan" Yi
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from image_processor import ImageProcessor

def process_image(image, filter_type, brightness_factor=1.0, kernel_size=5, flip_direction="horizontal"):
    """
    Process the uploaded image with the selected filter
    
    Args:
        image: Input image from Gradio
        filter_type: Selected filter type
        brightness_factor: Brightness adjustment factor
        kernel_size: Size of the Gaussian kernel
        flip_direction: Direction for image flipping ('horizontal' or 'vertical')
        
    Returns:
        Processed image
    """
    # Save the uploaded image to a temporary file
    # Get the file extension from the input image
    file_extension = os.path.splitext(image)[1] if isinstance(image, str) else '.png'
    with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
        temp_path = temp_file.name
        # Convert RGB to BGR for OpenCV
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            # If it's a file path, just copy it
            import shutil
            shutil.copy2(image, temp_path)
    
    try:
        # Create ImageProcessor instance
        processor = ImageProcessor(temp_path)
        
        # Apply selected filter
        if filter_type == "Grayscale":
            processed = processor.grayscale_conversion()
        elif filter_type == "Brightness":
            processed = processor.adjust_brightness(brightness_factor)
        elif filter_type == "Gaussian Blur":
            processed = processor.gaussian_blur(kernel_size)
        elif filter_type == "Cartoonize":
            processed = processor.cartoonize()
        elif filter_type == "Sepia":
            processed = processor.sepia()
        elif filter_type == "Flip":
            processed = processor.flip_image(flip_direction)
        else:
            processed = processor.image
        
        # Convert BGR to RGB for display
        if len(processed.shape) == 2:  # Grayscale image
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
        return processed
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

# Custom CSS for styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding: 20px;
    background-color: #f8f9fa;
}

.title {
    text-align: center;
    font-size: 2.5em !important;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 30px;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.upload-box {
    border: 2px dashed #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    background: white;
    transition: all 0.3s ease;
}

.upload-box:hover {
    border-color: #0066ff;
    background: #f8f9ff;
}

.control-group {
    margin-bottom: 15px;
}

.control-label {
    font-weight: 500;
    color: #1a1a1a;
    margin-bottom: 8px;
}

.process-btn {
    background: #0066ff !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.process-btn:hover {
    background: #0052cc !important;
    transform: translateY(-1px);
}

.output-image {
    border-radius: 12px;
    overflow: hidden;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.footer {
    text-align: center;
    color: #666;
    font-size: 0.9em;
    margin-top: 30px;
    padding: 20px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    # Title
    gr.Markdown("""
    <div class="title">
        🖼️ AI Image Filter Studio
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input card
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Upload Image")
                input_image = gr.Image(
                    label="",
                    elem_classes="upload-box",
                    interactive=True
                )
                
                # Controls
                with gr.Group(elem_classes="control-group"):
                    filter_type = gr.Dropdown(
                        choices=["Grayscale", "Brightness", "Gaussian Blur", 
                                "Cartoonize", "Sepia", "Flip"],
                        label="Select Filter",
                        elem_classes="control-label"
                    )
                    
                    brightness_factor = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Brightness Factor",
                        visible=False,
                        elem_classes="control-group"
                    )
                    
                    kernel_size = gr.Slider(
                        minimum=3,
                        maximum=15,
                        value=5,
                        step=2,
                        label="Kernel Size",
                        visible=False,
                        elem_classes="control-group"
                    )
                    
                    flip_direction = gr.Dropdown(
                        choices=["horizontal", "vertical"],
                        label="Flip Direction",
                        visible=False,
                        elem_classes="control-group"
                    )
                
                submit_btn = gr.Button(
                    "Process Image",
                    elem_classes="process-btn"
                )
        
        with gr.Column(scale=1):
            # Output card
            with gr.Group(elem_classes="card"):
                gr.Markdown("### Processed Image")
                output_image = gr.Image(
                    label="",
                    elem_classes="output-image"
                )
    
    # Footer
    gr.Markdown("""
    <div class="footer">
        Created with by Ziming "Alan" Yi
    </div>
    """)
    
    # Show/hide controls based on filter selection
    def toggle_controls(filter_type):
        brightness_visible = filter_type == "Brightness"
        kernel_visible = filter_type == "Gaussian Blur"
        flip_visible = filter_type == "Flip"
        return {
            brightness_factor: gr.update(visible=brightness_visible),
            kernel_size: gr.update(visible=kernel_visible),
            flip_direction: gr.update(visible=flip_visible)
        }
    
    filter_type.change(
        fn=toggle_controls,
        inputs=[filter_type],
        outputs=[brightness_factor, kernel_size, flip_direction]
    )
    
    # Process image when submit button is clicked
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, filter_type, brightness_factor, kernel_size, flip_direction],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch() 