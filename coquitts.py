from fastapi import FastAPI
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

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Cria o cliente Gradio
    client = Client("https://coquitts.nandus.com.br/")
    
    # Executa a predição usando os dados da requisição
    result = client.predict(
        data.text_prompt,  # str  in 'Texto do Prompt'
        data.language,  # str in 'Idioma'
        data.audio_reference,  # str in 'Áudio de Referência'
        data.use_microphone,  # str in 'Use microfone para referência'
        data.clean_voice,  # bool in 'Limpar Voz de Referência'
        data.no_auto_detect,  # bool in 'Não usar detecção automática de idioma'
        data.agree,  # bool in 'Concordo'
        fn_index=1
    )
    
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
