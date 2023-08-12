import gradio as gr
import json
import re
from data.data_process import DocumentProcessor
from models.model_loader import ModelLoader,QASystem
from utils.custom_logger import logger

# Load config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

logger.info(f"Loaded config file: {config}")

# Loading embedding model
document_processor = DocumentProcessor(model_name=config["embedding_model_name"], chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])

# Load model globally
model_loder = ModelLoader(config["model_id"], config["max_length"], config["temperature"],config['load_int8'])
llm = model_loder.load_model()

qa_system = QASystem(llm)

# Initialize global variable for doc_embedding
doc_embedding = None
pdf_file_name = None  
qa = None
def chatbot(pdf_file,query):
    global doc_embedding
    global pdf_file_name
    global qa
    if pdf_file_name is None or pdf_file_name!= pdf_file.name or doc_embedding is None:
        logger.info("New PDF Found Resetting doc_embedding")
        doc_embedding = None
        pdf_file_name = pdf_file.name
    if doc_embedding is None:
        logger.info("Starting for new doc_embedding")
        doc_embedding = document_processor.process_document(pdf_file.name)
        qa = qa_system.setup_retrieval_qa(doc_embedding)
    result = qa({"query": query})
    return re.sub(r'\n+', '\n', result['result'])

with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as demo:
    gr.Markdown("# Ask your Question to PDF Document")
    with gr.Row():
        with gr.Column(scale=4):
            pdf_file = gr.File(label="Upload your PDF") 
    output = gr.Textbox(label="output",lines=3)
    query = gr.Textbox(label="query")
    btn = gr.Button("Submit")
    btn.click(fn=chatbot, inputs=[pdf_file,query], outputs=[output])
gr.close_all()
demo.launch(share=True)