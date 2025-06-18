import hashlib
import gradio as gr
import os
from PIL import Image
import requests
import base64

os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('./tmp')

# Sample preloaded example images
API_URL = "http://localhost:8314/predict"
EXAMPLE_IMAGES = [
    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/cheetah1.jpg",
]

def api_run(image, model_number):
    """API call function for different models"""
    if image is None:
        return "Please upload an image first."
    
    # Map model number to model name
    model_mapping = {
        1: "clara",
        2: "gemini",
        3: "gpt"
    }
    
    model_name = model_mapping.get(model_number, "clara")
    
    os.makedirs("output", exist_ok=True)
    # Hash image content
    hash_image = hashlib.md5(image.tobytes()).hexdigest()
    image.save(os.path.join("output", f"{hash_image}.png"))
    
    # Convert image to base64
    with open(f"output/{hash_image}.png", "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Prepare request payload
    payload = {
        "images": encoded_image,
        "text": "Ảnh X-quang này có gì bất thường?",
        "model_name": model_name
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()["outputs"]
        else:
            return f"❌ Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

def load_example_image(selection: gr.SelectData):
    """Load example image from gallery"""
    return EXAMPLE_IMAGES[selection.index]

css = """
.output-box {
    border: 2px solid #888;
    border-radius: 10px;
    padding: 15px;
    background-color: #1e1e2f;  /* nền tối nhẹ nếu giao diện dark mode */
    margin-top: 10px;
}
"""


# Create the Gradio interface
with gr.Blocks(title="Multi-Modal AI Assistant", theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🤖 Multi-Modal AI Assistant")
    gr.Markdown("Upload an image and get responses from different AI models!")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image upload box
            image_input = gr.Image(
                label="📷 Upload Image", 
                type="pil",
                height=300
            )
            
            # Example images section
            gr.Markdown("### 🖼️ Example Images")
            gr.Markdown("Click any example below to load it:")
            
            example_gallery = gr.Gallery(
                value=EXAMPLE_IMAGES,
                label="Select an example image",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=1,
                object_fit="cover",
                height="auto",
                allow_preview=False,
                selected_index=None
            )
        
        with gr.Column(scale=2):
            # Tabs for different models
            # with gr.Tabs():
            #     with gr.TabItem("🚀 Clara Model"):
            #         output_1 = gr.Textbox(
            #             label="Model Response",
            #             placeholder="Response will appear here...",
            #             lines=10,
            #             interactive=False
            #         )
            #         submit_1 = gr.Button("Send", scale=1, variant="primary")
                
            #     with gr.TabItem("Gemini"):
            #         output_2 = gr.Textbox(
            #             label="Model Response",
            #             placeholder="Response will appear here...",
            #             lines=10,
            #             interactive=False
            #         )
            #         submit_2 = gr.Button("Send", scale=1, variant="primary")
                
            #     with gr.TabItem("ChatGPT"):   
            #         output_3 = gr.Textbox(
            #             label="Model Response",
            #             placeholder="Response will appear here...",
            #             lines=10,
            #             interactive=False
            #         )
            #         submit_3 = gr.Button("Send", scale=1, variant="primary")
            with gr.Tabs():
                with gr.TabItem("🚀 Clara Model"):
                    output_1 = gr.Markdown(
                        label="Model Response",
                        value="",
                        show_label=True,
                        elem_id="clara-output",
                        elem_classes=["output-box"]

                    )
                    submit_1 = gr.Button("Send", scale=1, variant="primary")
                
                with gr.TabItem("Gemini"):
                    output_2 = gr.Markdown(
                        label="Model Response",
                        value="",
                        show_label=True,
                        elem_id="gemini-output",
                        elem_classes=["output-box"]

                    )
                    submit_2 = gr.Button("Send", scale=1, variant="primary")
                
                with gr.TabItem("ChatGPT"):   
                    output_3 = gr.Markdown(
                        label="Model Response",
                        value="",
                        show_label=True,
                        elem_id="chatgpt-output",
                        elem_classes=["output-box"]

                    )
                    submit_3 = gr.Button("Send", scale=1, variant="primary")

    # Event handlers for example gallery
    example_gallery.select(
        fn=load_example_image,
        outputs=image_input
    )

    # Event handlers for Model 1
    submit_1.click(
        fn=lambda img: api_run(img, 1),
        inputs=[image_input],
        outputs=output_1
    )
    

    # Event handlers for Model 2
    submit_2.click(
        fn=lambda img: api_run(img, 2),
        inputs=[image_input],
        outputs=output_2
    )
    
    # Event handlers for Model 3
    submit_3.click(
        fn=lambda img: api_run(img,3),
        inputs=[image_input],
        outputs=output_3
    )
    
if __name__ == "__main__":
    demo.launch(share=True, debug=True)