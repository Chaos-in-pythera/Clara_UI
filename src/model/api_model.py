import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# Load .env once
load_dotenv()


# === GEMINI PIPELINE ===
class GeminiMedicalPipeline:
    def __init__(self, model_name="gemini-2.0-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def build_prompt(self, user_instruction):
        return f"""
Bạn là một trợ lý chẩn đoán hình ảnh y khoa chuyên nghiệp.

Dựa trên ảnh y tế và câu hỏi của người dùng dưới đây, hãy phân tích và trả lời **bằng định dạng markdown** như sau:

## 📌 Dấu hiệu nổi bật
- Gạch đầu dòng mô tả chi tiết bất thường trên ảnh

## 🩺 Nhận định sơ bộ
- Một đoạn tóm tắt mang tính chuyên môn, gợi ý chẩn đoán

Hãy giữ văn phong y khoa rõ ràng, không phỏng đoán nếu không có cơ sở rõ ràng từ hình ảnh.

--- CÂU HỎI CỦA NGƯỜI DÙNG ---
{user_instruction}
"""

    def run(self, image: Image.Image, user_instruction: str) -> str:
        prompt = self.build_prompt(user_instruction)
        response = self.model.generate_content([prompt, image])
        return response.text


# === OPENROUTER (ChatGPT/Gemini/GPT-4-Vision) PIPELINE ===
class ChatGPTMedicalVisionPipeline:
    def __init__(self, model="openai/o4-mini"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model

    def image_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_text_instruction(self, user_instruction: str) -> str:
        return f"""
Bạn là một trợ lý chẩn đoán hình ảnh y khoa chuyên nghiệp.

Hãy xem xét ảnh y tế bên dưới và phân tích **bằng định dạng markdown** như sau:

## 📌 Dấu hiệu nổi bật
- Liệt kê các bất thường đáng chú ý trên ảnh

## 🩺 Nhận định sơ bộ
- Đưa ra nhận định hoặc chẩn đoán sơ bộ dựa trên các dấu hiệu đã quan sát được

⚠️ Không được suy đoán nếu hình ảnh không đủ rõ ràng.

--- Câu hỏi ---
{user_instruction}
"""

    def run(self, image: Image.Image, user_instruction: str) -> str:
        image_base64 = self.image_to_base64(image)

        response = self.client.chat.completions.create(
            model=self.model,
            extra_headers={
                "HTTP-Referer": "https://yourdomain.com",  # tùy chọn
                "X-Title": "Medical Assistant"
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.build_text_instruction(user_instruction)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )

        return response.choices[0].message.content
    
    
if __name__ == "__main__":
    image_path = "/home/truongnn/chaos/code/repo/medical_inferneces/examples/test_1.png"
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    question = "Ảnh X-quang phổi này có gì bất thường? Hãy phân tích và kết luận sơ bộ dưới dạng markdown."

    # Dùng Gemini
    gemini_pipeline = GeminiMedicalPipeline()
    gemini_result = gemini_pipeline.run(image, question)
    print("📊 Gemini Result:\n", gemini_result)

    # Dùng OpenRouter (gpt-4-vision hoặc gemini-pro-vision)
    chatgpt_pipeline = ChatGPTMedicalVisionPipeline(model="openai/o4-mini")
    chatgpt_result = chatgpt_pipeline.run(image, question)
    print("📊 OpenRouter Result:\n", chatgpt_result)

