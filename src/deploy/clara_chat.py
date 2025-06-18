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

EXAMPLE_IMAGES_DICT = [
    {'images_1': './examples/sample/test_1.png', 'message':'Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 53 tuổi. Cho biết bệnh nhân bị gì?'},
    {'images_2': './examples/sample/test_2.png', 'message':'Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 35 tuổi. Cho biết bệnh nhân bị gì?'},
    {'images_3': './examples/sample/test_3.png', 'message':'Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 61 tuổi. Cho biết bệnh nhân bị gì?'},
    {'images_4': './examples/sample/test_4.png', 'message':' Ảnh chụp X-ray PA (Chụp Xquang tim phổi thẳng) bệnh nhân nam, 79 tuổi. Cho biết bệnh nhân bị gì?'},
    {'images_5': './examples/sample/test_5.png', 'message':'Chụp Xquang tim phổi thẳng) bệnh nhân nữ, 33 tuổi. Cho biết bệnh nhân bị gì?'},
    
]

def api_run(image, message, history, model_number):
    """Model processing function with model selection"""
    if image is None:
        return history + [["Please upload an image first.", None]]
    
    # Map model number to model name
    model_mapping = {
        1: "clara",
        2: "gemini",
        3: "gpt"
    }
    
    model_name = model_mapping.get(model_number, "clara")  # Default to clara if invalid number
    
    os.makedirs("output", exist_ok=True)
    hash_image = hashlib.md5(image.tobytes()).hexdigest()
    image.save(os.path.join("output", f"{hash_image}.png"))
    
    encoded_image = None
    with open(f"output/{hash_image}.png", "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
        
    payload = {
        "images": encoded_image,
        "text": message,
        "model_name": model_name
    }
    
    response = requests.post(API_URL, json=payload)
    text = response.json()["outputs"]
    
    if response.status_code == 200:
        print(f"🧠 {model_name.upper()} Output:\n", text)
    else:
        print("❌ Error:", response.status_code, text)
    
    history.append([message, text])
    return history


def load_example_image(example_path):
    """Load example image and return it"""
    return example_path

def clear_chat():
    """Clear chat history"""
    return []

# Create the Gradio interface
# with gr.Blocks(title="Multi-Modal AI Assistant", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("# 🤖 Multi-Modal AI Assistant")
#     gr.Markdown("Upload an image and chat with different AI models!")
    
#     with gr.Row():
#         with gr.Column(scale=1):
#             # Image upload box
#             image_input = gr.Image(
#                 label="📷 Upload Image", 
#                 type="pil",
#                 height=300
#             )
            
#             # Example images section
#             gr.Markdown("### 🖼️ Example Images")
#             gr.Markdown("Click any example below to load it:")
            
#             example_gallery = gr.Gallery(
#                 value=EXAMPLE_IMAGES,
#                 label="Select an example image",
#                 show_label=True,
#                 elem_id="gallery",
#                 columns=2,
#                 rows=1,
#                 object_fit="cover",
#                 height="auto",
#                 allow_preview=False,
#                 selected_index=None
#             )
        
#         with gr.Column(scale=2):
#             # Tabs for different models
#             with gr.Tabs():
#                 with gr.TabItem("🚀 Clara Model"):
#                     chatbot_1 = gr.Chatbot(label="Chat with Clara", height=400, render_markdown= True)
#                     with gr.Row():
#                         msg_1 = gr.Textbox(
#                             label="Message", 
#                             placeholder="Ask something about the image...",
#                             scale=4
#                         )
#                         submit_1 = gr.Button("Send", scale=1, variant="primary")
#                         clear_1 = gr.Button("Clear", scale=1)
                
#                 with gr.TabItem("Gemini"):
#                     chatbot_2 = gr.Chatbot(label="Chat with Gemini", height=400, render_markdown= True)
#                     with gr.Row():
#                         msg_2 = gr.Textbox(
#                             label="Message", 
#                             placeholder="What would you like to analyze?",
#                             scale=4
#                         )
#                         submit_2 = gr.Button("Send", scale=1, variant="primary")
#                         clear_2 = gr.Button("Clear", scale=1)
                
#                 with gr.TabItem("ChatGPT"):
#                     chatbot_3 = gr.Chatbot(label="Chat with ChatGPT", height=400, render_markdown= True)
#                     with gr.Row():
#                         msg_3 = gr.Textbox(
#                             label="Message", 
#                             placeholder="How can I help you with this image?",
#                             scale=4
#                         )
#                         submit_3 = gr.Button("Send", scale=1, variant="primary")
#                         clear_3 = gr.Button("Clear", scale=1)

#     # Event handlers for example buttons
#     def show_warning(selection: gr.SelectData):
#         return EXAMPLE_IMAGES[selection.index]

#     example_gallery.select(
#         fn=show_warning,
#         outputs=image_input
#     )
with gr.Blocks(title="Multi-Modal AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Modal AI Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image input and examples section
            image_input = gr.Image(
                label="📷 Upload Image", 
                type="pil",
                height=300
            )
            
            gr.Markdown("### 🖼️ Example Cases")
            gr.Markdown("Click any example below to load it:")
            example_gallery = gr.Gallery(
            value = [
            next(v for k, v in example.items() if k.startswith("images_"))
            for example in EXAMPLE_IMAGES_DICT
        ],
                label="Click an example to load image and message",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=3,
                height="auto",
                allow_preview=False
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                # Define all UI elements first
                with gr.TabItem("🚀 Clara Vision"):
                    chatbot_1 = gr.Chatbot(label="Chat with Clara", height=400, render_markdown=True)
                    with gr.Row():
                        msg_1 = gr.Textbox(
                            label="Message", 
                            placeholder="Ask something about the image...",
                            scale=4
                        )
                        submit_1 = gr.Button("Send", scale=1, variant="primary")
                        clear_1 = gr.Button("Clear", scale=1)
                
                with gr.TabItem("🔬 Gemini Vision"):
                    chatbot_2 = gr.Chatbot(label="Chat with Gemini", height=400, render_markdown=True)
                    with gr.Row():
                        msg_2 = gr.Textbox(
                            label="Message", 
                            placeholder="What would you like to analyze?",
                            scale=4
                        )
                        submit_2 = gr.Button("Send", scale=1, variant="primary")
                        clear_2 = gr.Button("Clear", scale=1)
                
                with gr.TabItem("🎯 ChatGPT Vision"):
                    chatbot_3 = gr.Chatbot(label="Chat with ChatGPT", height=400, render_markdown=True)
                    with gr.Row():
                        msg_3 = gr.Textbox(
                            label="Message", 
                            placeholder="How can I help you with this image?",
                            scale=4
                        )
                        submit_3 = gr.Button("Send", scale=1, variant="primary")
                        clear_3 = gr.Button("Clear", scale=1)

    # After all UI elements are defined, set up the example gallery handler
    def load_example(evt: gr.SelectData):
        """Load example image and its corresponding message"""
        selected_example = EXAMPLE_IMAGES_DICT[evt.index]
        image_path = selected_example[f'images_{evt.index + 1}']
        message = selected_example['message']
        return [
            image_path,  # Load image
            message,     # Fill message in all tabs
            message,
            message
        ]

    # Connect the example gallery event handler
    example_gallery.select(
        fn=load_example,
        outputs=[
            image_input,
            msg_1,
            msg_2,
            msg_3
        ]
    )

    submit_1.click(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 1),
        inputs=[image_input, msg_1, chatbot_1],
        outputs=chatbot_1
    ).then(
        fn=lambda: "",
        outputs=msg_1
    )
    
    msg_1.submit(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 1),
        inputs=[image_input, msg_1, chatbot_1],
        outputs=chatbot_1
    ).then(
        fn=lambda: "",
        outputs=msg_1
    )

    # Event handlers for Model 2 (Gemini)
    submit_2.click(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 2),
        inputs=[image_input, msg_2, chatbot_2],
        outputs=chatbot_2
    ).then(
        fn=lambda: "",
        outputs=msg_2
    )
    
    msg_2.submit(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 2),
        inputs=[image_input, msg_2, chatbot_2],
        outputs=chatbot_2
    ).then(
        fn=lambda: "",
        outputs=msg_2
    )

    # Event handlers for Model 3 (ChatGPT)
    submit_3.click(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 3),
        inputs=[image_input, msg_3, chatbot_3],
        outputs=chatbot_3
    ).then(
        fn=lambda: "",
        outputs=msg_3
    )
    
    msg_3.submit(
        fn=lambda img, msg, hist: api_run(img, msg, hist, 3),
        inputs=[image_input, msg_3, chatbot_3],
        outputs=chatbot_3
    ).then(
        fn=lambda: "",
        outputs=msg_3
    )
    clear_3.click(fn=clear_chat, outputs=chatbot_3)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)