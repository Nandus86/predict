from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client

class InputData(BaseModel):
    text_prompt: str
    language: str
    audio_reference: str
    use_microphone: str
    clean_voice: bool
    no_auto_detect: bool
    agree: bool
    fn_index: int
    usar_mic: str  # Adicionando parâmetro adicional

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    client = Client("https://coquitts.nandus.com.br/")
    
    # Ajuste a chamada conforme os parâmetros esperados
    result = client.predict(
        (data.text_prompt, data.language),  # Supondo que a API espera um tuplo
        (data.audio_reference, data.use_microphone),
        data.clean_voice,
        data.no_auto_detect,
        data.agree,
        data.fn_index,
        data.usar_mic
    )
    
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
