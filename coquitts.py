from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

class DadosEntrada(BaseModel):
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

# Atualizar a URL do cliente Gradio
cliente = Client("https://coquitts.nandus.com.br/")

@app.post("/predict")
async def prever(dados: DadosEntrada):
    try:
        resultado = cliente.predict(
            dados.text_prompt,
            dados.language,
            dados.audio_reference,
            dados.use_microphone,
            dados.usar_mic,
            dados.clean_voice,
            dados.no_auto_detect,
            dados.agree,
            fn_index=dados.fn_index
        )
        return {"resultado": resultado}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
