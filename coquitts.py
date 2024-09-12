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
    usar_mic: str

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    # Cria o cliente Gradio
    client = Client("https://coquitts.nandus.com.br/")

    # Executa a predição usando os dados da requisição
    try:
        # Especifique o `fn_index` e os argumentos conforme esperado pela API Gradio
        result = client.predict(
            api_name=None,  # Se necessário, ajuste para o nome do endpoint
            fn_index=data.fn_index,  # Índice da função
            inputs=(
                data.text_prompt,  # str
                data.language,  # str
                data.audio_reference,  # str
                data.use_microphone,  # str
                data.clean_voice,  # bool
                data.no_auto_detect,  # bool
                data.agree,  # bool
                data.usar_mic  # str
            )
        )
        return {"result": result}
    except Exception as e:
        # Captura e retorna qualquer erro
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
