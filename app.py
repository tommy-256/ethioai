import gradio as gr  
from transformers import AutoModelForCausalLM, AutoTokenizer  

# Load the model and tokenizer  
MODEL_NAME = "thatstommy/lora_model"  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  

def generate(text):  
    inputs = tokenizer(text, return_tensors="pt")  # Tokenize the input  
    outputs = model.generate(**inputs)  # Generate text using the model  
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output  

# Custom CSS for styling  
css = """  
/* Change the background color */  
#component-0 {  
    background-color: #e0e0e0; /* Light gray background */  
}  

/* Change the color of the title */  
#title {  
   font-size: 24px;  
   color: #003366; /* Dark blue */  
   text-align: center;  
}  

/* Change the color of output text */  
.output-text {  
   font-size: 22px;  
   color: #004080; /* Slightly lighter dark blue */  
   text-align: center;  
}  

/* Add logo styling */  
#logo {  
    display: block;  
    margin: 0 auto;  
    width: 150px; /* Size of the logo */  
}  

/* Input and output styling */  
.input-textbox, .output-textbox {  
    border-radius: 5px;  
    border: 1px solid #003366; /* Dark blue border */  
    background-color: #ffffff; /* White background for input/output */  
}  
"""  

# Create the Gradio interface with Blocks  
with gr.Blocks(css=css) as interface:  
    gr.Markdown("<h1 id='title'>Your Brand Name</h1>")  # Brand Name  
    gr.Image(value="C:\\Users\\tommy\\Downloads\\logo.jpg", label="Logo", elem_id="logo")  # Logo  
    input_text = gr.Textbox(label="Enter your prompt:", placeholder="Type your prompt here...")  
    output_text = gr.Textbox(label="Output:", interactive=False)  

    # Button to trigger the model generation function  
    btn = gr.Button("Generate")  

    # On button click, link the input to the model function and output  
    btn.click(fn=generate, inputs=input_text, outputs=output_text)  

# Launch the interface  
if __name__ == "__main__":  
    interface.launch(server_name="0.0.0.0", server_port=7860, show_error=False, prevent_thread_lock=True)