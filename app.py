from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name = 'Fine-tuned-model-codeT5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)

    # Generate code using the model
    outputs = model.generate(inputs['input_ids'], max_length=4096,)    

    # Decode the generated code
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template('index.html', prompt=prompt, generated_code=generated_code)

if __name__ == '__main__':
    app.run(debug=True)
