from fastapi import FastAPI, HTTPException
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
    usar_mic: str

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    client = Client("https://coquitts.nandus.com.br/")
    
    try:
        # Substitua '1' pelo fn_index correto
        result = client.predict(
            api_name=None,  # Ou o nome do endpoint se necessário
            fn_index=data.fn_index,  # Índice da função
            inputs=[data.text_prompt, data.language, data.audio_reference, data.use_microphone, data.clean_voice, data.no_auto_detect, data.agree, data.usar_mic]
        )
        return {"result": result}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
