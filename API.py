# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import fitz  # PyMuPDF for PDF processing
import openai  # or any other LLM provider

app = FastAPI()

# Store uploaded PDFs text in memory
pdf_knowledge_base = {}

# Helper function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as pdf:
        for page_num in range(pdf.page_count):
            page = pdf.load_page(page_num)
            text += page.get_text("text")
    return text

# Endpoint to upload PDFs
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    pdf_knowledge_base[file.filename] = text
    return {"filename": file.filename, "message": "PDF uploaded successfully"}

# Request schema for chat messages
class ChatRequest(BaseModel):
    question: str
    filename: str

# Endpoint to handle chat requests
@app.post("/ask_question/")
async def ask_question(request: ChatRequest):
    # Retrieve document text
    document_text = pdf_knowledge_base.get(request.filename)
    if not document_text:
        return JSONResponse(content={"error": "PDF not found"}, status_code=404)
    
    # Use OpenAI or any LLM to get an answer based on document content
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Answer the following question based on this document:\n\n{document_text}\n\nQuestion: {request.question}",
        max_tokens=150
    )
    answer = response.choices[0].text.strip()
    
    return {"answer": answer}
