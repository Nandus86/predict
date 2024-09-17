from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from gradio_client import Client
import os
import shutil

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

        wav_file_path = [r for r in result if r.endswith('.wav')][0]
        final_directory = "/app/final_audio/"
        os.makedirs(final_directory, exist_ok=True)
        final_wav_path = shutil.move(wav_file_path, final_directory)
        temp_dir = os.path.dirname(wav_file_path)
        shutil.rmtree(temp_dir)  # Apaga diretórios temporários

        return {"result": final_wav_path}
    except Exception as e:
        return {"error": str(e)}

# Rota para servir os arquivos .wav
@app.get("/audio/{file_name}")
async def get_audio(file_name: str):
    file_path = f"/app/final_audio/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='audio/wav')
    else:
        return {"error": "Arquivo não encontrado"}

# Rota para apagar arquivos .wav
@app.delete("/audio/{file_name}")
async def delete_audio(file_name: str):
    file_path = f"/app/final_audio/{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Arquivo {file_name} foi apagado com sucesso."}
    else:
        return {"error": "Arquivo não encontrado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
