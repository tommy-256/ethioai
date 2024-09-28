import gradio as gr  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  # Import torch for device management  

# Load the model and tokenizer  
MODEL_NAME = "thatstommy/lora_model"  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  

# Move model to appropriate device  
device = "cuda" if torch.cuda.is_available() else "cpu"  
model.to(device)  

def generate(text):  
    try:  
        inputs = tokenizer(text, return_tensors="pt").to(device)  # Tokenize the input and move to device  
        outputs = model.generate(**inputs)  # Generate text using the model  
        return tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output  
    except Exception as e:  
        return f"An error occurred: {str(e)}"  

# Custom CSS for styling  
css = """  
#title {  
    text-align: center;  
    margin-bottom: 20px;  
}  
#input-text {  
    width: 100%;  
}  
#output-text {  
    width: 100%;  
}  
#component-0 {  
    background-color: #e0e0e0; /* Light gray background */  
    padding: 20px;  
    border-radius: 10px;  
}  
"""  

# Create the Gradio interface with Blocks  
with gr.Blocks(css=css) as interface:  
    gr.Markdown("<h1 id='title'>Your Brand Name</h1>")  # Brand Name  
    input_text = gr.Textbox(label="Enter your prompt:", placeholder="Type your prompt here...", elem_id="input-text")  
    output_text = gr.Textbox(label="Output:", interactive=False, elem_id="output-text")  

    # Button to trigger the model generation function  
    btn = gr.Button("Generate")  

    # On button click, link the input to the model function and output  
    btn.click(fn=generate, inputs=input_text, outputs=output_text)  

# Launch the interface  
if __name__ == "__main__":  
    interface.launch()
