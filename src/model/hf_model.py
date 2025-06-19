

from unsloth import FastLanguageModel
from PIL import Image
from transformers import TextStreamer
import torch

class ClaraPipeline:
    def __init__(self, model_path, max_seq_length=1024, max_tokens= 512):
        self.max_seq_length = max_seq_length
        self.max_tokens= max_tokens
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            # load_in_8bit= False,
            # dtype= torch.bfloat16
        )
        FastLanguageModel.for_inference(self.model)  # enable 2x inference speed


    def _generate_response(self, conversation, image):
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            image,
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        generated_ids = self.model.generate(
            **inputs, streamer=streamer, max_new_tokens=self.max_tokens
        )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def run(self, image, follow_up_question):
        conversation = []

        # Step 1: First question with image
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": follow_up_question},
                {"type": "image", "image": image}
            ]
        })
        response_1 = self._generate_response(conversation, image)
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": response_1}]})

        # Step 2: Follow-up question
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": 'Kết luận từ thông tin đó bệnh nhân bị gì?, Hãy nói chi tiết.'}]
        })
        response_2 = self._generate_response(conversation, image)
        output = f"##  Nhận xét hình ảnh:\n\n{response_1}\n\n---\n\n##  Kết luận:\n\n{response_2}"

        return output
