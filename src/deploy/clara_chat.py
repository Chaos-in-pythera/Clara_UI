import hashlib
import gradio as gr
import os
from PIL import Image
import requests
import base64
os.environ['GRADIO_TEMP_DIR'] = os.path.expanduser('./tmp')


# Sample preloaded example images (you can replace with actual image paths)
API_URL = "http://localhost:8314/predict"  # hoặc IP nếu deploy từ xa
EXAMPLE_IMAGES = [
    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/cheetah1.jpg",
]



# Mock model functions for each tab
def api_run(image, message, history):
    """Mock Model 1 processing function"""
    if image is None:
        return history + [["Please upload an image first.", None]]
    
    os.makedirs("output", exist_ok=True)
    # hash image content
    hash_image = hashlib.md5(image.tobytes()).hexdigest()
    image.save(os.path.join("output", f"{hash_image}.png"))
    # open image and convert to base64
    encoded_image = None
    with open(f"output/{hash_image}.png", "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
    # Step 2: Prepare request payload
    payload = {
        "images": encoded_image,
        "text": message,
        "model_name": "clara"
    }
    # Step 4: Print response
    response = requests.post(API_URL, json=payload)
    text = response.json()["outputs"]
    if response.status_code == 200:
        print("🧠 Model Output:\n", text)
    else:
        print("❌ Error:", response.status_code, text)
    print(f"Model 1 analyzed your image and message: '{message}'. I can see an image with dimensions {image.size}.")
    history.append([message, text])
    return history


def load_example_image(example_path):
    """Load example image and return it"""
    return example_path

def clear_chat():
    """Clear chat history"""
    return []

# Create the Gradio interface
with gr.Blocks(title="Multi-Modal AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Modal AI Assistant")
    gr.Markdown("Upload an image and chat with different AI models!")
    
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
            with gr.Tabs():
                with gr.TabItem("🚀 Model 1 - Vision GPT"):
                    chatbot_1 = gr.Chatbot(label="Chat with Model 1", height=400)
                    with gr.Row():
                        msg_1 = gr.Textbox(
                            label="Message", 
                            placeholder="Ask something about the image...",
                            scale=4
                        )
                        submit_1 = gr.Button("Send", scale=1, variant="primary")
                        clear_1 = gr.Button("Clear", scale=1)
                
                with gr.TabItem("🔬 Model 2 - Image Analyzer"):
                    chatbot_2 = gr.Chatbot(label="Chat with Model 2", height=400)
                    with gr.Row():
                        msg_2 = gr.Textbox(
                            label="Message", 
                            placeholder="What would you like to analyze?",
                            scale=4
                        )
                        submit_2 = gr.Button("Send", scale=1, variant="primary")
                        clear_2 = gr.Button("Clear", scale=1)
                
                with gr.TabItem("🎯 Model 3 - Advanced AI"):
                    chatbot_3 = gr.Chatbot(label="Chat with Model 3", height=400)
                    with gr.Row():
                        msg_3 = gr.Textbox(
                            label="Message", 
                            placeholder="How can I help you with this image?",
                            scale=4
                        )
                        submit_3 = gr.Button("Send", scale=1, variant="primary")
                        clear_3 = gr.Button("Clear", scale=1)

    # Event handlers for example buttons
    def show_warning(selection: gr.SelectData):
        return EXAMPLE_IMAGES[selection.index]

    example_gallery.select(
        fn=show_warning,
        outputs=image_input
    )

    # Event handlers for Model 1
    submit_1.click(
        fn=api_run,
        inputs=[image_input, msg_1, chatbot_1],
        outputs=chatbot_1
    ).then(
        fn=lambda: "",
        outputs=msg_1
    )
    
    msg_1.submit(
        fn=api_run,
        inputs=[image_input, msg_1, chatbot_1],
        outputs=chatbot_1
    ).then(
        fn=lambda: "",
        outputs=msg_1
    )
    
    clear_1.click(fn=clear_chat, outputs=chatbot_1)

    # Event handlers for Model 2
    submit_2.click(
        fn=api_run,
        inputs=[image_input, msg_2, chatbot_2],
        outputs=chatbot_2
    ).then(
        fn=lambda: "",
        outputs=msg_2
    )
    
    msg_2.submit(
        fn=api_run,
        inputs=[image_input, msg_2, chatbot_2],
        outputs=chatbot_2
    ).then(
        fn=lambda: "",
        outputs=msg_2
    )
    
    clear_2.click(fn=clear_chat, outputs=chatbot_2)

    # Event handlers for Model 3
    submit_3.click(
        fn=api_run,
        inputs=[image_input, msg_3, chatbot_3],
        outputs=chatbot_3
    ).then(
        fn=lambda: "",
        outputs=msg_3
    )
    
    msg_3.submit(
        fn=api_run,
        inputs=[image_input, msg_3, chatbot_3],
        outputs=chatbot_3
    ).then(
        fn=lambda: "",
        outputs=msg_3
    )
    
    clear_3.click(fn=clear_chat, outputs=chatbot_3)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)