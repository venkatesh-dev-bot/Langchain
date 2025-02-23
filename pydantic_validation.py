from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import dotenv
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Load environment variables (if needed)
dotenv.load_dotenv()

app = FastAPI()

# Define input models
class Question(BaseModel):
    query: str
    model: str  # New field to choose the model

class Country(BaseModel):
    name: str
    state: List[str] = Field(default_factory=list)
    capital: Optional[List[str]] = None  # Accept list of lists of strings


# Define prompt and parser
pydantic_parser = PydanticOutputParser(pydantic_object=Country)
format_instructions = pydantic_parser.get_format_instructions()

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chat API. Use the /ask endpoint to interact."}

@app.post("/ask")
async def ask_question(question: Question):
    # Select the appropriate language model based on user choice
    if question.model == "gpt":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif question.model == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_retries=2
        )
    elif question.model == "mistral":
        llm = ChatMistralAI(
            model="codestral-latest",
            temperature=0,
            max_retries=2,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model selection. Choose 'gpt', 'gemini', or 'mistral'.")

    # Define the prompt using the selected LLM
    prompt = PromptTemplate(
        template="You are an informative assistant. Please provide the following information for the query: {user_input}\n{format_instructions}\n",
        input_variables=["user_input"],
        partial_variables={"format_instructions": format_instructions},
    )
    
    # Create the LLM chain
    chain = prompt | llm | pydantic_parser 

    # Invoke the chain with user's query
    response = await chain.ainvoke(question.query)
    return JSONResponse(content=response.model_dump(), status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
