from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

# Define o modelo de entrada para o FastAPI
class InputData(BaseModel):
    text_prompt: str
    language: str
    audio_reference: str
    use_microphone: str
    clean_voice: bool
    no_auto_detect: bool
    agree: bool
    fn_index: int
    usar_mic: bool

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Cria o cliente Gradio
    client = Client("https://coquitts.nandus.com.br/")
    
    try:
        # Chama o método predict da API Gradio passando os parâmetros diretamente
        result = client.predict(
            data.text_prompt,         # str
            data.language,            # str
            data.audio_reference,    # str
            data.use_microphone,     # str
            data.clean_voice,        # bool
            data.no_auto_detect,     # bool
            data.agree,              # bool
            data.fn_index,           # int
            data.usar_mic            # bool
        )
        return {"result": result}
    except ValueError as e:
        # Captura erros específicos relacionados a parâmetros
        raise HTTPException(status_code=400, detail=f"ValueError: {e}")
    except Exception as e:
        # Captura outros erros
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
