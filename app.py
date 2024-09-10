import gradio as gr  
from transformers import AutoModelForCausalLM, AutoTokenizer  

# Replace 'your-username/your-model-name' with your actual Hugging Face model repo  
MODEL_NAME = "thatstommy/lora_model"  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  

def generate(text):  
    inputs = tokenizer(text, return_tensors="pt")  
    outputs = model.generate(**inputs)  
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  

interface = gr.Interface(fn=generate, inputs="text", outputs="text")  
interface.launch()