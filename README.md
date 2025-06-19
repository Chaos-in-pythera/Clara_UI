# Clara UI - Multi-Modal Medical Image Analysis Assistant

A medical image analysis system that combines multiple AI models (Clara, Gemini, and ChatGPT) to provide comprehensive analysis of medical images, particularly focused on X-ray interpretations.

## 🌟 Features

- Support for multiple AI models:
  - Clara: pythera-trained medical image analysis model
  - Gemini: Google's Gemini model
  - ChatGPT: OpenAI's model
- Interactive web interface using Gradio
- Support for X-ray image analysis
- Markdown-formatted responses
- Example gallery with sample medical cases
- RESTful API service

## 🛠️ Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Required API keys:
  - Google API key for Gemini
  - OpenRouter API key for ChatGPT integration

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Chaos-in-pythera/Clara_UI.git
cd Clara_UI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 🚀 Running the Application

1. Start the API server:
```bash
python -m src/api/main_api.py
```

2. Launch the UI (in a separate terminal):
```bash
python -m src/deploy/clara_fix.py
```

The application will be available at `http://localhost:7860`

## 🏗️ Project Structure

```
Clara_UI/
├── src/
│   ├── api/
│   │   └── main_api.py         # FastAPI backend
│   ├── deploy/
│   │   ├── clara_chat.py       # Chat interface
│   │   └── clara_fix.py        # Fixed interface
│   └── model/
│       ├── api_model.py        # Gemini & ChatGPT models
│       └── hf_model.py         # Clara model
├── examples/
│   └── sample/                 # Example medical images
└── .env                        # Environment variables
```

## 🔧 Configuration

- Model paths and configurations can be adjusted in `src/api/main_api.py`
- UI customization can be done in the Gradio interface files
- Example cases can be modified in the `EXAMPLE_IMAGES_DICT` in the UI files

## 📄 License

[Add your license information here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
