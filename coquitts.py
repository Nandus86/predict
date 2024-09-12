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

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Cria o cliente Gradio
    client = Client("https://coquitts.nandus.com.br/")
    
    try:
        # Executa a predição usando os dados da requisição
        result = client.predict(
            data.text_prompt,  # str  in 'Texto do Prompt'
            data.language,  # str in 'Idioma'
            data.audio_reference,  # str in 'Áudio de Referência'
            data.use_microphone,  # str in 'Use microfone para referência'
            data.clean_voice,  # bool in 'Limpar Voz de Referência'
            data.no_auto_detect,  # bool in 'Não usar detecção automática de idioma'
            data.agree,  # bool in 'Concordo'
            data.fn_index  # Garanta que fn_index esteja correto
        )
        return {"result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"ValueError: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
