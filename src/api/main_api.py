from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import base64
from ..model.hf_model import ClaraPipeline
from ..model.api_model import GeminiMedicalPipeline, ChatGPTMedicalVisionPipeline

app = FastAPI()


class PredictionRequest(BaseModel):
    images: str  # Base64 encoded image
    text: str
    model_name: str

class PredictionResponse(BaseModel):
    outputs: str
    
# init clara model 
model_path = '/home/truongnn/chaos/code/repo/medical_inferneces/model_hf_cached'

# clara_model = ClaraPipeline(model_path)
clara_model = None
gemini_pipeline = GeminiMedicalPipeline()
chat_gpt_pipeline = ChatGPTMedicalVisionPipeline()



@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global clara_model
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.images)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((448, 448))
        
    
        
        if request.model_name.lower() == 'clara':
            response = clara_model.run(image, request.text)
            return PredictionResponse(outputs=response)
        
        elif request.model_name.lower() == 'gemini':
            response = gemini_pipeline.run(image, request.text)
            return PredictionResponse(outputs=response)
        
        elif request.model_name.lower() == 'gpt':
            response = chat_gpt_pipeline.run(image, request.text)
            return PredictionResponse(outputs=response)
            

        # Add other model types here
        # elif isinstance(model, OtherModel):
        #     response = model.different_predict_method(image, request.text)
        else:
            raise ValueError(f"Unsupported model type: {type(request.model_name)}")
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8314)