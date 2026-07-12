# 📄 Kagaz.ai (কাগজ): The Bureaucratic Un-Complicator

**Kagaz.ai** is a pro-bono civic assistant designed to cut through bureaucratic jargon for Indian citizens. Built for the Gemma for Bharat Hackathon (Productivity & Daily Life track).

When citizens receive official municipality notices, bank default letters, or legal summons, they are often written in dense legalese. Kagaz.ai allows users to upload a photo of these documents and outputs a plain-language summary, the document type, a 1-2-3 action plan, and an urgency level in regional languages like Bengali and Hindi.

## 🛠️ Tech Stack
* **Model:** Google Gemma 4 Vision (`gemma-4-e2b-it`)
* **Hardware:** Kaggle T4 GPU (Local edge-inference for data sovereignty)
* **Frontend:** Gradio Blocks

## 🚀 How to Run
This application is designed to run in a Kaggle Notebook environment. 
1. Open a Kaggle Notebook with a T4 GPU enabled.
2. Install dependencies: `!pip install gradio accelerate kagglehub`
3. Run the `app.py` script to launch the local web server.
