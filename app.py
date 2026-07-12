import torch
import gradio as gr
import kagglehub
from transformers import AutoProcessor, AutoModelForCausalLM

# --- 1. Load the Model ---
print("Downloading Gemma 4 weights for Kagaz.ai...")
model_handle = "google/gemma-4/transformers/gemma-4-e2b-it"
model_path = kagglehub.model_download(model_handle)

print("Loading model into Kaggle T4 GPU...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)
print("Kagaz.ai Civic Engine is live!")

# --- 2. The Core Simplification Logic ---
def simplify_document(input_image, language_preference):
    if input_image is None:
        return "Please upload a photo of the official document or letter."

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"You are Kagaz.ai, a pro-bono civic assistant for Indian citizens. Analyze the uploaded photo of this official, legal, or bureaucratic document.\n\nCut through the legal jargon and provide your response in {language_preference} using this exact structure:\n\n**1. Document Type**: (What is this document? e.g., Municipal Tax Notice, Bank Default, etc.)\n**2. Plain Language Summary**: (Explain what this document says in simple, everyday language as if explaining it to a worried citizen. Remove all bureaucratic jargon.)\n**3. Action Plan**: (List a simple 1-2-3 checklist of what the person needs to do next based on this document.)\n**4. Urgency Level**: (Is this Low, Medium, or High urgency? Why?)\n\nBe highly empathetic but strictly factual. Do not provide binding legal advice."}
                ]
            }
        ]

        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = processor(
            text=prompt_text, 
            images=input_image, 
            return_tensors="pt"
        ).to("cuda", torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.2)

        generated_text = outputs[0][inputs["input_ids"].shape[-1]:]
        response = processor.decode(generated_text, skip_special_tokens=True)

        return response
        
    except Exception as e:
        return f"⚠️ An error occurred during AI processing: {str(e)}\n\nPlease try a clearer or slightly smaller image."

# --- 3. The Web Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Kagaz.ai (কাগজ): The Bureaucratic Un-Complicator")
    gr.Markdown("Official letters and legal notices are terrifying because they are hard to read. Upload a photo of any dense government, bank, or legal document. The Gemma 4 engine will strip away the jargon and tell you exactly what it means and what to do next.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Official Document (Photo)")
            lang_dropdown = gr.Dropdown(
                choices=["English", "Bengali", "Hindi"], 
                value="Bengali", 
                label="Output Language"
            )
            with gr.Row():
                clear_btn = gr.ClearButton(components=[image_input], value="Clear Form", variant="secondary")
                submit_btn = gr.Button("Un-Complicate Document", variant="primary")
            
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Kagaz.ai Plain-Language Summary", lines=15)

    submit_btn.click(fn=simplify_document, inputs=[image_input, lang_dropdown], outputs=text_output)
    
    gr.Markdown("---")
    gr.Markdown("⚠️ **Disclaimer:** *Kagaz.ai is a hackathon prototype designed for educational purposes. It utilizes AI to simplify text but does not provide legally binding advice. Always consult a qualified legal professional for severe or urgent legal matters.*")

demo.queue()
demo.launch(share=True, theme=gr.themes.Base())
