import os
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI()

client = Client("https://coquitts.nandus.com.br/")

class PredictRequest(BaseModel):
    text_prompt: str
    language: str
    audio_reference: str
    use_microphone: str
    clean_voice: bool
    no_auto_detect: bool
    agree: bool
    fn_index: int
    usar_mic: bool

@app.post("/predict")
async def predict(request_data: PredictRequest):
    try:
        # Chama o Gradio Client para fazer a predição
        result = client.predict(
            request_data.text_prompt,
            request_data.language,
            request_data.audio_reference,
            request_data.use_microphone,
            request_data.usar_mic,
            request_data.clean_voice,
            request_data.no_auto_detect,
            request_data.agree,
            fn_index=request_data.fn_index
        )

        # Filtra o arquivo .wav do resultado
        wav_file_path = [r for r in result if r.endswith('.wav')][0]

        # Define o diretório de destino para o arquivo final
        final_directory = "/app/final_audio/"
        os.makedirs(final_directory, exist_ok=True)

        # Move o arquivo .wav para o diretório final e apaga temporários
        final_wav_path = shutil.move(wav_file_path, final_directory)
        temp_dir = os.path.dirname(wav_file_path)
        shutil.rmtree(temp_dir)  # Apaga o diretório temporário

        return {"result": final_wav_path}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
