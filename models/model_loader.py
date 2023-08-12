from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
from utils.custom_logger import logger

class ModelLoader:
    """
    A class responsible for loading the language model.
    """

    def __init__(self, model_id, max_length, temperature,load_int8):
        """
        Initialize the ModelLoader instance.

        Args:
            model_id (str): Identifier of the pretrained model.
            max_length (int): Maximum length of generated text.
            temperature (float): Temperature parameter for text generation.
        """
        self.model_id = model_id
        self.max_length = max_length
        self.temperature = temperature
        self.load_int8 = load_int8
        
    def load_model(self):
        """
        Load the language model using the specified model_id, max_length, and temperature.

        Returns:
            HuggingFacePipeline: Loaded language model.
        """
        logger.info(f"Loading LLM model {self.model_id} with max_length {self.max_length} and temperature {self.temperature}...\n")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.load_int8:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_8bit=True, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16, device_map="auto")
        
        logger.info("Model is loaded successfully\n")
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_length=self.max_length, temperature=self.temperature
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

class QASystem:
    """
    A class representing a Question Answering (QA) system.
    """

    def __init__(self, llm):
        """
        Initialize the QASystem instance.

        Args:
            llm (HuggingFacePipeline): Loaded language model for text generation.
        """
        self.llm = llm

        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}

        Question: {question}
        Answer :"""
        PROMPT = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )
        self.chain_type_kwargs = {
            "prompt": PROMPT,
        }

    def setup_retrieval_qa(self, doc_embedding):
        """
        Set up the retrieval-based QA system.

        Args:
            doc_embedding: Document embedding for retrieval.

        Returns:
            RetrievalQA: Configured retrieval-based QA system.
        """
        logger.info("Setting up retrieval QA system...\n")
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # You might need to replace this with the appropriate chain type.
            retriever=doc_embedding.as_retriever(),
            chain_type_kwargs=self.chain_type_kwargs,
        )

        return qa
