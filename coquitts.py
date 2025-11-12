from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client
import whisper
import torch
import soundfile as sf
import os
import shutil
import glob
import tempfile
import requests
import struct
from typing import Optional

app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente será inicializado quando necessário
client = None

# Carrega o modelo Whisper na inicialização
print("Carregando modelo Whisper...")
whisper_model = whisper.load_model("base")
print("Modelo Whisper carregado com sucesso!")

# ============== MODELOS PYDANTIC ==============

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
    hash_folder: str

class TranscriptionRequest(BaseModel):
    hash_folder: str
    language: Optional[str] = None

class TranscriptionFromUrlRequest(BaseModel):
    audio_url: str
    hash_folder: str = "default"
    language: Optional[str] = None
    model_size: Optional[str] = "base"

class GTTSRequest(BaseModel):
    text: str
    hash_folder: str = "default"
    speed: float = 1.0

class GenaiTTSRequest(BaseModel):
    text: str
    hash_folder: str = "default"
    voice_name: str = "Leda"

# ============== FUNÇÕES UTILITÁRIAS ==============

def get_next_sequential_filename(directory: str, extension: str = ".wav") -> str:
    """Gera o próximo nome de arquivo sequencial na pasta"""
    if not os.path.exists(directory):
        return f"1{extension}"
    
    try:
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        if not files:
            return f"1{extension}"
        
        numbers = []
        for filename in files:
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext.isdigit():
                numbers.append(int(name_without_ext))
        
        if not numbers:
            return f"1{extension}"
        
        return f"{max(numbers) + 1}{extension}"
    except Exception:
        return f"1{extension}"

def get_gradio_client():
    """Inicializa cliente Gradio sob demanda"""
    global client
    if client is None:
        try:
            client = Client("https://coquitts.nandus.com.br/")
            print("Cliente Gradio conectado com sucesso!")
        except Exception as e:
            print(f"Erro ao conectar cliente Gradio: {e}")
            raise HTTPException(status_code=503, detail="Serviço TTS temporariamente indisponível")
    return client

def parse_audio_mime_type(mime_type: str) -> dict:
    """Parseia bits per sample e rate de audio MIME type"""
    bits_per_sample = 16
    rate = 24000
    
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Converte audio data para WAV com header correto"""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

# ============== ENDPOINTS TTS ==============

@app.post("/predict")
async def predict(request_data: PredictRequest):
    try:
        print(f"Iniciando predição para hash: {request_data.hash_folder}")
        
        gradio_client = get_gradio_client()
        
        result = gradio_client.predict(
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
        
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        shutil.move(wav_file_path, final_wav_path)
        
        temp_dir = os.path.dirname(wav_file_path)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        
        print(f"Predição concluída: {sequential_filename}")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename
        }
        
    except Exception as e:
        print(f"Erro na predição: {str(e)}")
        return {"error": str(e)}

@app.post("/gtts")
async def google_tts_generate(request_data: GTTSRequest):
    """Gera áudio usando Google Text-to-Speech"""
    try:
        print(f"Iniciando gTTS para hash: {request_data.hash_folder}")
        
        try:
            from gtts import gTTS
        except ImportError:
            raise HTTPException(status_code=503, detail="Instale: pip install gtts")
        
        if len(request_data.text) > 1000:
            raise HTTPException(status_code=400, detail="Texto muito longo. Máximo 1000 caracteres.")
        
        print(f"Processando: '{request_data.text[:100]}...'")
        
        tts = gTTS(
            text=request_data.text,
            lang='pt',
            slow=False if request_data.speed >= 1.0 else True,
            tld='com.br'
        )
        
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        temp_mp3 = final_wav_path.replace('.wav', '_temp.mp3')
        tts.save(temp_mp3)
        
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3)
            
            if request_data.speed != 1.0:
                new_sample_rate = int(audio.frame_rate * request_data.speed)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                audio = audio.set_frame_rate(audio.frame_rate)
            
            audio.export(final_wav_path, format="wav")
            sample_rate = audio.frame_rate
            duration = len(audio) / 1000.0
            
        except ImportError:
            shutil.move(temp_mp3, final_wav_path)
            sample_rate = 22050
            duration = 0
            print("Aviso: pydub não instalado - conversão limitada")
        
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        
        print(f"gTTS concluído: {sequential_filename}")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename,
            "model": "google-tts",
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "text_length": len(request_data.text)
        }
        
    except Exception as e:
        print(f"Erro no gTTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/genaitts")
async def genai_tts_generate(request_data: GenaiTTSRequest):
    """Gera áudio usando Google Gemini 2.5 Flash TTS"""
    try:
        print(f"Iniciando Gemini TTS para hash: {request_data.hash_folder}")
        
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise HTTPException(status_code=503, detail="Instale Google Genai: pip install google-genai")
        
        if len(request_data.text) > 1000:
            raise HTTPException(status_code=400, detail="Texto muito longo. Máximo 1000 caracteres.")
        
        print(f"Processando: '{request_data.text[:100]}...'")
        print(f"Voz: {request_data.voice_name}")
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY não configurada")
        
        client = genai.Client(api_key=api_key)
        
        model = "gemini-2.5-flash-preview-tts"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=request_data.text),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=request_data.voice_name
                    )
                )
            ),
        )
        
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        audio_chunks = []
        
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            
            if (chunk.candidates[0].content.parts[0].inline_data and 
                chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                mime_type = inline_data.mime_type
                
                if not mime_type.startswith("audio/wav"):
                    data_buffer = convert_to_wav(data_buffer, mime_type)
                
                audio_chunks.append(data_buffer)
        
        if audio_chunks:
            with open(final_wav_path, "wb") as f:
                for chunk in audio_chunks:
                    f.write(chunk)
        else:
            raise HTTPException(status_code=500, detail="Nenhum áudio gerado pelo Gemini")
        
        file_size = os.path.getsize(final_wav_path)
        sample_rate = 24000
        bytes_per_sample = 2
        duration = file_size / (sample_rate * bytes_per_sample)
        
        print(f"Gemini TTS concluído: {sequential_filename} ({duration:.2f}s)")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename,
            "model": "gemini-2.5-flash-tts",
            "voice": request_data.voice_name,
            "duration_seconds": duration,
            "text_length": len(request_data.text)
        }
        
    except Exception as e:
        print(f"Erro no Gemini TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== ENDPOINTS WHISPER TRANSCRIPTION ==============

@app.post("/transcriptions")
async def transcribe_audio(
    audio: UploadFile = File(...),
    hash_folder: str = "default",
    language: Optional[str] = None
):
    """Transcreve áudio usando Whisper"""
    try:
        print(f"Iniciando transcrição para hash: {hash_folder}")
        
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser um áudio")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            if language:
                result = whisper_model.transcribe(
                    temp_file_path, 
                    language=language,
                    fp16=False,
                    verbose=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0
                )
            else:
                result = whisper_model.transcribe(
                    temp_file_path,
                    fp16=False,
                    verbose=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0
                )
            
            transcription_directory = f"/app/transcriptions/{hash_folder}/"
            os.makedirs(transcription_directory, exist_ok=True)
            
            txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
            transcription_path = os.path.join(transcription_directory, txt_filename)
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            print(f"Transcrição concluída: {txt_filename}")
            
            return {
                "text": result["text"],
                "language": result["language"],
                "hash_folder": hash_folder,
                "filename": txt_filename,
                "transcription_path": transcription_path
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcriptions/from-url")
async def transcribe_from_url(request_data: TranscriptionFromUrlRequest):
    """Transcreve áudio a partir de uma URL"""
    try:
        print(f"Baixando áudio de: {request_data.audio_url}")
        
        response = requests.get(request_data.audio_url, stream=True, timeout=30)
        response.raise_for_status()
        
        if request_data.audio_url.endswith('.oga'):
            ext = '.oga'
        elif request_data.audio_url.endswith('.mp3'):
            ext = '.mp3'
        elif request_data.audio_url.endswith('.wav'):
            ext = '.wav'
        else:
            content_type = response.headers.get('content-type', '')
            if 'ogg' in content_type:
                ext = '.oga'
            elif 'mpeg' in content_type or 'mp3' in content_type:
                ext = '.mp3'
            else:
                ext = '.oga'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        print(f"Áudio baixado. Iniciando transcrição...")
        
        try:
            if request_data.language:
                result = whisper_model.transcribe(
                    temp_file_path, 
                    language=request_data.language,
                    fp16=False,
                    verbose=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0
                )
            else:
                result = whisper_model.transcribe(
                    temp_file_path,
                    fp16=False,
                    verbose=False,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0
                )
            
            transcription_directory = f"/app/transcriptions/{request_data.hash_folder}/"
            os.makedirs(transcription_directory, exist_ok=True)
            
            txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
            transcription_path = os.path.join(transcription_directory, txt_filename)
            
            with open(transcription_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            print(f"Transcrição de URL concluída: {txt_filename}")
            
            return {
                "text": result["text"],
                "language": result["language"],
                "source_url": request_data.audio_url,
                "hash_folder": request_data.hash_folder,
                "filename": txt_filename,
                "transcription_path": transcription_path
            }
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except requests.RequestException as e:
        print(f"Erro ao baixar áudio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao baixar áudio: {str(e)}")
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcriptions/from-url-form")
async def transcribe_from_url_form(
    audio_url: str = Form(...),
    hash_folder: str = Form(default="default"),
    language: Optional[str] = Form(default=None)
):
    """Transcreve áudio a partir de uma URL usando form-data"""
    request_data = TranscriptionFromUrlRequest(
        audio_url=audio_url,
        hash_folder=hash_folder,
        language=language
    )
    return await transcribe_from_url(request_data)

@app.post("/transcriptions/from-file")
async def transcribe_from_existing_file(request_data: TranscriptionRequest):
    """Transcreve um arquivo de áudio já existente no servidor"""
    try:
        print(f"Transcrevendo arquivo existente para hash: {request_data.hash_folder}")
        
        audio_directory = f"/app/final_audio/{request_data.hash_folder}/"
        
        if not os.path.exists(audio_directory):
            raise HTTPException(status_code=404, detail="Pasta de áudio não encontrada")
        
        audio_files = [f for f in os.listdir(audio_directory) if f.endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        
        if not audio_files:
            raise HTTPException(status_code=404, detail="Nenhum arquivo de áudio encontrado")
        
        audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
        latest_audio = audio_files[-1]
        audio_file_path = os.path.join(audio_directory, latest_audio)
        
        if request_data.language:
            result = whisper_model.transcribe(
                audio_file_path, 
                language=request_data.language,
                fp16=False,
                verbose=False,
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
        else:
            result = whisper_model.transcribe(
                audio_file_path,
                fp16=False,
                verbose=False,
                beam_size=1,
                best_of=1,
                temperature=0.0
            )
        
        transcription_directory = f"/app/transcriptions/{request_data.hash_folder}/"
        os.makedirs(transcription_directory, exist_ok=True)
        
        txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
        transcription_path = os.path.join(transcription_directory, txt_filename)
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        print(f"Transcrição concluída: {txt_filename}")
        
        return {
            "text": result["text"],
            "language": result["language"],
            "source_audio": latest_audio,
            "hash_folder": request_data.hash_folder,
            "filename": txt_filename,
            "transcription_path": transcription_path
        }
        
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== ENDPOINTS DE GERENCIAMENTO ==============

@app.get("/audio/{hash_folder}/{file_name}")
async def get_audio(hash_folder: str, file_name: str):
    file_path = f"/app/final_audio/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type='audio/wav',
            headers={
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes"
            }
        )
    else:
        return {"error": "Arquivo não encontrado"}

@app.get("/transcriptions/{hash_folder}/{file_name}")
async def get_transcription(hash_folder: str, file_name: str):
    file_path = f"/app/transcriptions/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type='text/plain',
            headers={
                "Cache-Control": "public, max-age=3600"
            }
        )
    else:
        return {"error": "Arquivo de transcrição não encontrado"}

@app.delete("/audio/{hash_folder}/{file_name}")
async def delete_audio(hash_folder: str, file_name: str):
    file_path = f"/app/final_audio/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Arquivo {file_name} foi apagado com sucesso."}
    else:
        return {"error": "Arquivo não encontrado"}

@app.delete("/transcriptions/{hash_folder}/{file_name}")
async def delete_transcription(hash_folder: str, file_name: str):
    file_path = f"/app/transcriptions/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Transcrição {file_name} foi apagada com sucesso."}
    else:
        return {"error": "Arquivo de transcrição não encontrado"}

@app.get("/audio/{hash_folder}")
async def list_audio_files(hash_folder: str):
    directory = f"/app/final_audio/{hash_folder}/"
    if not os.path.exists(directory):
        return {"error": "Pasta não encontrada"}
    
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
    
    return {"hash_folder": hash_folder, "files": files}

@app.get("/transcriptions/{hash_folder}")
async def list_transcription_files(hash_folder: str):
    directory = f"/app/transcriptions/{hash_folder}/"
    if not os.path.exists(directory):
        return {"error": "Pasta de transcrições não encontrada"}
    
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
    
    return {"hash_folder": hash_folder, "files": files}

@app.delete("/audio/{hash_folder}")
async def delete_hash_folder(hash_folder: str):
    directory = f"/app/final_audio/{hash_folder}/"
    if os.path.exists(directory):
        shutil.rmtree(directory)
        return {"message": f"Pasta de áudio {hash_folder} foi apagada com sucesso."}
    else:
        return {"error": "Pasta não encontrada"}

@app.delete("/transcriptions/{hash_folder}")
async def delete_transcription_folder(hash_folder: str):
    directory = f"/app/transcriptions/{hash_folder}/"
    if os.path.exists(directory):
        shutil.rmtree(directory)
        return {"message": f"Pasta de transcrições {hash_folder} foi apagada com sucesso."}
    else:
        return {"error": "Pasta de transcrições não encontrada"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
