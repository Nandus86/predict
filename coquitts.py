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
import psutil  # Para controle de CPU
import threading
from typing import Optional

app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

# Cliente será inicializado quando necessário
client = None

# Carrega o modelo Whisper na inicialização (mais eficiente)
print("Carregando modelo Whisper...")
whisper_model = whisper.load_model("base")  # Melhor balance velocidade/precisão
print("Modelo Whisper carregado com sucesso!")

# Carrega Parler-TTS na inicialização
parler_model = None
parler_tokenizer = None

# Carrega Parler-TTS na inicialização (VERSÃO COMPLETA)
parler_model = None
parler_tokenizer = None

# Controle de CPU
def limit_cpu_usage():
    """Limita uso de CPU para 80%"""
    current_process = psutil.Process()
    current_process.nice(10)  # Reduz prioridade
    
    # Limita threads do PyTorch
    torch.set_num_threads(max(1, psutil.cpu_count() - 1))

# Inicializa Parler-TTS automaticamente (com configurações otimizadas)
try:
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    
    print("Carregando Parler-TTS...")
    limit_cpu_usage()  # Aplica limite de CPU
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "freds0/parler-tts-mini-v1.1-ptbr"
    
    # Carrega modelo
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    ).to(device)
    
    parler_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Parler-TTS carregado com sucesso! Dispositivo: {device}")
    print(f"Threads PyTorch limitadas a: {torch.get_num_threads()}")
    
except Exception as e:
    print(f"Parler-TTS não carregado: {e}")
    parler_model = None
    parler_tokenizer = None

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
    hash_folder: str  # Nova propriedade para o hash da pasta

class TranscriptionRequest(BaseModel):
    hash_folder: str
    language: Optional[str] = None  # Se não especificado, auto-detecta

class TranscriptionFromUrlRequest(BaseModel):
    audio_url: str
    hash_folder: str = "default"
    language: Optional[str] = None
    model_size: Optional[str] = "base"  # tiny, base, small

class ParlerTTSRequest(BaseModel):
    text: str
    description: str = "Uma voz feminina jovem e clara falando em português brasileiro"
    hash_folder: str = "default"
    speed: float = 1.0
    temperature: float = 1.0
    do_sample: bool = True
    max_length_multiplier: float = 2.0  # Multiplicador do tamanho do texto
    early_stopping: bool = True

# ============== FUNÇÕES UTILITÁRIAS ==============

def get_next_sequential_filename(directory: str, extension: str = ".wav") -> str:
    """
    Gera o próximo nome de arquivo sequencial na pasta (otimizado)
    """
    if not os.path.exists(directory):
        return f"1{extension}"
    
    # Usa listdir que é mais rápido que glob para este caso
    try:
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        if not files:
            return f"1{extension}"
        
        # Extrai números de forma otimizada
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

def get_parler_model():
    """Retorna modelos Parler-TTS já carregados"""
    global parler_model, parler_tokenizer
    if parler_model is None:
        raise HTTPException(status_code=503, detail="Parler-TTS não foi carregado na inicialização")
    return parler_model, parler_tokenizer

def move_file_sync(src: str, dst: str) -> None:
    """Operação síncrona de mover arquivo"""
    shutil.move(src, dst)

# ============== ENDPOINTS TTS (Original) ==============

@app.post("/predict")
async def predict(request_data: PredictRequest):
    try:
        print(f"Iniciando predição para hash: {request_data.hash_folder}")
        
        # Inicializa cliente se necessário
        gradio_client = get_gradio_client()
        
        # Executa a predição (simplificado)
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
        
        # Cria o diretório baseado no hash
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        # Gera o próximo nome sequencial (otimizado)
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        # Move o arquivo para o destino final
        shutil.move(wav_file_path, final_wav_path)
        
        # Limpeza do diretório temporário
        temp_dir = os.path.dirname(wav_file_path)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass  # Se der erro na limpeza, ignora
        
        print(f"Predição concluída: {sequential_filename}")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename
        }
        
    except Exception as e:
        print(f"Erro na predição: {str(e)}")
        return {"error": str(e)}

@app.post("/parler-tts")
async def parler_tts_generate(request_data: ParlerTTSRequest):
    """
    Gera áudio usando Parler-TTS (modelo brasileiro local) - VERSÃO COMPLETA
    """
    try:
        print(f"Iniciando Parler-TTS para hash: {request_data.hash_folder}")
        print(f"Configurações: speed={request_data.speed}, temp={request_data.temperature}")
        
        # Verifica se modelo está carregado
        if parler_model is None:
            raise HTTPException(status_code=503, detail="Parler-TTS não carregado")
        
        # Limita tamanho do texto
        if len(request_data.text) > 500:
            raise HTTPException(status_code=400, detail="Texto muito longo. Máximo 500 caracteres.")
        
        print(f"Processando: '{request_data.text[:100]}...'")
        print(f"Voz: '{request_data.description[:100]}...'")
        
        device = next(parler_model.parameters()).device
        
        # Prepara inputs com configurações completas
        try:
            input_ids = parler_tokenizer(
                request_data.description, 
                return_tensors="pt", 
                max_length=512,
                truncation=True,
                padding=True
            ).input_ids.to(device)
            
            prompt_input_ids = parler_tokenizer(
                request_data.text, 
                return_tensors="pt",
                max_length=300,
                truncation=True,
                padding=True
            ).input_ids.to(device)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro no tokenizer: {str(e)}")
        
        print("Gerando áudio com configurações personalizadas...")
        
        # Calcula max_length baseado no texto
        base_length = prompt_input_ids.shape[-1]
        max_gen_length = int(base_length * request_data.max_length_multiplier)
        max_gen_length = min(max_gen_length, 1024)  # Limite absoluto
        
        # Gera áudio com todas as configurações
        try:
            with torch.no_grad():
                generation = parler_model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    do_sample=request_data.do_sample,
                    temperature=request_data.temperature,
                    max_length=max_gen_length,
                    pad_token_id=parler_tokenizer.pad_token_id,
                    eos_token_id=parler_tokenizer.eos_token_id,
                    early_stopping=request_data.early_stopping,
                    num_return_sequences=1,
                    repetition_penalty=1.1,  # Evita repetições
                    length_penalty=1.0,      # Controla tamanho
                    no_repeat_ngram_size=3   # Evita loops
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro na geração: {str(e)}")
        
        # Processa áudio
        try:
            audio_arr = generation.cpu().numpy().squeeze()
            if len(audio_arr.shape) > 1:
                audio_arr = audio_arr[0]
            
            if len(audio_arr) == 0:
                raise Exception("Áudio vazio gerado")
            
            # Sample rate do modelo
            sample_rate = getattr(parler_model.config, 'sampling_rate', 22050)
            
            print(f"Áudio base gerado: {len(audio_arr)} samples, {sample_rate}Hz")
            
            # Aplica ajuste de velocidade se necessário
            if request_data.speed != 1.0:
                try:
                    import librosa
                    audio_arr = librosa.effects.time_stretch(audio_arr, rate=request_data.speed)
                    print(f"Velocidade ajustada para: {request_data.speed}x")
                except ImportError:
                    print("Librosa não disponível - velocidade ignorada")
                except Exception as e:
                    print(f"Erro no ajuste de velocidade: {e}")
            
            # Normaliza áudio
            if abs(audio_arr).max() > 0:
                audio_arr = audio_arr / abs(audio_arr).max() * 0.95  # 95% do máximo
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
        
        # Salva arquivo
        try:
            hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
            os.makedirs(hash_directory, exist_ok=True)
            
            sequential_filename = get_next_sequential_filename(hash_directory)
            final_wav_path = os.path.join(hash_directory, sequential_filename)
            
            sf.write(final_wav_path, audio_arr, sample_rate)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao salvar: {str(e)}")
        
        duration = len(audio_arr) / sample_rate
        print(f"Parler-TTS concluído: {sequential_filename} ({duration:.2f}s)")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename,
            "model": "parler-tts-ptbr-full",
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "text_length": len(request_data.text),
            "settings": {
                "speed": request_data.speed,
                "temperature": request_data.temperature,
                "do_sample": request_data.do_sample,
                "max_length": max_gen_length,
                "description": request_data.description[:100]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Erro geral no Parler-TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/gtts")
async def google_tts_generate(request_data: ParlerTTSRequest):
    """
    Gera áudio usando Google Text-to-Speech (RÁPIDO - 2-5 segundos)
    """
    try:
        print(f"Iniciando gTTS para hash: {request_data.hash_folder}")
        
        # Importa gTTS
        try:
            from gtts import gTTS
        except ImportError:
            raise HTTPException(status_code=503, detail="Instale: pip install gtts")
        
        # Limita tamanho do texto
        if len(request_data.text) > 1000:
            raise HTTPException(status_code=400, detail="Texto muito longo. Máximo 1000 caracteres.")
        
        print(f"Processando: '{request_data.text[:100]}...'")
        
        # Cria gTTS
        tts = gTTS(
            text=request_data.text,
            lang='pt',  # Português
            slow=False if request_data.speed >= 1.0 else True,
            tld='com.br'  # Sotaque brasileiro
        )
        
        # Cria diretório
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        # Gera nome sequencial
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        # Salva o áudio temporariamente como MP3
        temp_mp3 = final_wav_path.replace('.wav', '_temp.mp3')
        tts.save(temp_mp3)
        
        # Converte MP3 para WAV usando pydub (se disponível)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_mp3)
            
            # Aplica ajuste de velocidade se necessário
            if request_data.speed != 1.0:
                new_sample_rate = int(audio.frame_rate * request_data.speed)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                audio = audio.set_frame_rate(audio.frame_rate)
            
            audio.export(final_wav_path, format="wav")
            sample_rate = audio.frame_rate
            duration = len(audio) / 1000.0  # pydub usa milissegundos
            
        except ImportError:
            # Fallback: apenas renomeia o MP3 para WAV (não é ideal mas funciona)
            shutil.move(temp_mp3, final_wav_path)
            sample_rate = 22050  # Assume padrão do gTTS
            duration = 0  # Não consegue calcular sem pydub
            print("Aviso: pydub não instalado - conversão limitada")
        
        # Remove arquivo temporário se ainda existir
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
            "text_length": len(request_data.text),
            "settings": {
                "speed": request_data.speed,
                "language": "pt-BR",
                "engine": "google"
            }
        }
        
    except Exception as e:
        print(f"Erro no gTTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edge-tts")
async def edge_tts_generate(request_data: ParlerTTSRequest):
    """
    Gera áudio usando Microsoft Edge TTS (MUITO RÁPIDO - 1-3 segundos)
    """
    try:
        print(f"Iniciando Edge-TTS para hash: {request_data.hash_folder}")
        
        # Importa edge-tts
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise HTTPException(status_code=503, detail="Instale: pip install edge-tts")
        
        # Limita tamanho do texto
        if len(request_data.text) > 2000:
            raise HTTPException(status_code=400, detail="Texto muito longo. Máximo 2000 caracteres.")
        
        print(f"Processando: '{request_data.text[:100]}...'")
        
        # Escolhe voz baseada na descrição
        voice = "pt-BR-FranciscaNeural"  # Padrão feminino
        if "masculin" in request_data.description.lower() or "homem" in request_data.description.lower():
            voice = "pt-BR-AntonioNeural"
        elif "jovem" in request_data.description.lower():
            voice = "pt-BR-ThalitaNeural"
        
        # Ajusta velocidade (Edge-TTS usa porcentagem)
        rate = "+0%"
        if request_data.speed < 0.8:
            rate = "-20%"
        elif request_data.speed < 0.9:
            rate = "-10%"
        elif request_data.speed > 1.2:
            rate = "+20%"
        elif request_data.speed > 1.1:
            rate = "+10%"
        
        print(f"Voz selecionada: {voice}, Velocidade: {rate}")
        
        # Cria diretório
        hash_directory = f"/app/final_audio/{request_data.hash_folder}/"
        os.makedirs(hash_directory, exist_ok=True)
        
        # Gera nome sequencial
        sequential_filename = get_next_sequential_filename(hash_directory)
        final_wav_path = os.path.join(hash_directory, sequential_filename)
        
        # Cria comunicação Edge-TTS
        communicate = edge_tts.Communicate(
            request_data.text, 
            voice,
            rate=rate
        )
        
        # Gera e salva áudio
        await communicate.save(final_wav_path)
        
        # Calcula informações do arquivo
        try:
            import wave
            with wave.open(final_wav_path, 'r') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / float(sample_rate)
        except:
            sample_rate = 24000  # Padrão do Edge-TTS
            duration = len(request_data.text) * 0.1  # Estimativa
        
        print(f"Edge-TTS concluído: {sequential_filename}")
        
        return {
            "result": final_wav_path,
            "hash_folder": request_data.hash_folder,
            "filename": sequential_filename,
            "model": "microsoft-edge-tts",
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "text_length": len(request_data.text),
            "settings": {
                "voice": voice,
                "speed": request_data.speed,
                "rate": rate,
                "language": "pt-BR",
                "engine": "edge-tts"
            }
        }
        
    except Exception as e:
        print(f"Erro no Edge-TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============== ENDPOINTS WHISPER TRANSCRIPTION ==============

@app.post("/transcriptions")
async def transcribe_audio(
    audio: UploadFile = File(...),
    hash_folder: str = "default",
    language: Optional[str] = None
):
    """
    Transcreve áudio usando Whisper
    """
    try:
        print(f"Iniciando transcrição para hash: {hash_folder}")
        
        # Verifica se é um arquivo de áudio
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser um áudio")
        
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcreve com Whisper (otimizado para velocidade)
            if language:
                result = whisper_model.transcribe(
                    temp_file_path, 
                    language=language,
                    fp16=False,  # Força FP32 (evita warning)
                    verbose=False,  # Remove logs verbosos
                    beam_size=1,  # Reduz beam search (mais rápido)
                    best_of=1,  # Reduz tentativas (mais rápido)
                    temperature=0.0  # Determinístico (mais rápido)
                )
            else:
                result = whisper_model.transcribe(
                    temp_file_path,
                    fp16=False,  # Força FP32 (evita warning)
                    verbose=False,  # Remove logs verbosos
                    beam_size=1,  # Reduz beam search (mais rápido)
                    best_of=1,  # Reduz tentativas (mais rápido)
                    temperature=0.0  # Determinístico (mais rápido)
                )
            
            # Cria diretório para salvar a transcrição
            transcription_directory = f"/app/transcriptions/{hash_folder}/"
            os.makedirs(transcription_directory, exist_ok=True)
            
            # Gera nome sequencial para a transcrição
            txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
            transcription_path = os.path.join(transcription_directory, txt_filename)
            
            # Salva a transcrição em arquivo
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
            # Remove arquivo temporário
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcriptions/from-url")
async def transcribe_from_url(request_data: TranscriptionFromUrlRequest):
    """
    Transcreve áudio a partir de uma URL (mais rápido!)
    """
    try:
        print(f"Baixando áudio de: {request_data.audio_url}")
        
        # Baixa o arquivo de áudio
        response = requests.get(request_data.audio_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Detecta extensão do arquivo pela URL ou Content-Type
        if request_data.audio_url.endswith('.oga'):
            ext = '.oga'
        elif request_data.audio_url.endswith('.mp3'):
            ext = '.mp3'
        elif request_data.audio_url.endswith('.wav'):
            ext = '.wav'
        else:
            # Tenta detectar pelo Content-Type
            content_type = response.headers.get('content-type', '')
            if 'ogg' in content_type:
                ext = '.oga'
            elif 'mpeg' in content_type or 'mp3' in content_type:
                ext = '.mp3'
            else:
                ext = '.oga'  # Default para Telegram
        
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        print(f"Áudio baixado. Iniciando transcrição...")
        
        try:
            # Transcreve com Whisper (otimizado para velocidade)
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
            
            # Cria diretório para salvar a transcrição
            transcription_directory = f"/app/transcriptions/{request_data.hash_folder}/"
            os.makedirs(transcription_directory, exist_ok=True)
            
            # Gera nome sequencial para a transcrição
            txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
            transcription_path = os.path.join(transcription_directory, txt_filename)
            
            # Salva a transcrição em arquivo
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
            # Remove arquivo temporário
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
    """
    Transcreve áudio a partir de uma URL usando form-data
    """
    # Converte para o modelo Pydantic
    request_data = TranscriptionFromUrlRequest(
        audio_url=audio_url,
        hash_folder=hash_folder,
        language=language
    )
    
    # Reutiliza a lógica do endpoint JSON
    return await transcribe_from_url(request_data)

@app.post("/transcriptions/from-file")
async def transcribe_from_existing_file(request_data: TranscriptionRequest):
    """
    Transcreve um arquivo de áudio já existente no servidor
    """
    try:
        print(f"Transcrevendo arquivo existente para hash: {request_data.hash_folder}")
        
        # Procura por arquivos de áudio na pasta do hash
        audio_directory = f"/app/final_audio/{request_data.hash_folder}/"
        
        if not os.path.exists(audio_directory):
            raise HTTPException(status_code=404, detail="Pasta de áudio não encontrada")
        
        audio_files = [f for f in os.listdir(audio_directory) if f.endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        
        if not audio_files:
            raise HTTPException(status_code=404, detail="Nenhum arquivo de áudio encontrado na pasta")
        
        # Pega o último arquivo (mais recente numericamente)
        audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
        latest_audio = audio_files[-1]
        audio_file_path = os.path.join(audio_directory, latest_audio)
        
        # Transcreve com Whisper (otimizado)
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
        
        # Cria diretório para salvar a transcrição
        transcription_directory = f"/app/transcriptions/{request_data.hash_folder}/"
        os.makedirs(transcription_directory, exist_ok=True)
        
        # Gera nome sequencial para a transcrição
        txt_filename = get_next_sequential_filename(transcription_directory, ".txt")
        transcription_path = os.path.join(transcription_directory, txt_filename)
        
        # Salva a transcrição em arquivo
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        
        print(f"Transcrição do arquivo {latest_audio} concluída: {txt_filename}")
        
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

# ============== ENDPOINTS DE GERENCIAMENTO (Original + Transcrições) ==============

# Rota para servir os arquivos .wav (otimizada)
@app.get("/audio/{hash_folder}/{file_name}")
async def get_audio(hash_folder: str, file_name: str):
    file_path = f"/app/final_audio/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type='audio/wav',
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache por 1 hora
                "Accept-Ranges": "bytes"  # Suporte a streaming
            }
        )
    else:
        return {"error": "Arquivo não encontrado"}

# Rota para servir arquivos de transcrição
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

# Rota para apagar arquivos .wav específicos
@app.delete("/audio/{hash_folder}/{file_name}")
async def delete_audio(hash_folder: str, file_name: str):
    file_path = f"/app/final_audio/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Arquivo {file_name} foi apagado com sucesso."}
    else:
        return {"error": "Arquivo não encontrado"}

# Rota para apagar arquivos de transcrição específicos
@app.delete("/transcriptions/{hash_folder}/{file_name}")
async def delete_transcription(hash_folder: str, file_name: str):
    file_path = f"/app/transcriptions/{hash_folder}/{file_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Transcrição {file_name} foi apagada com sucesso."}
    else:
        return {"error": "Arquivo de transcrição não encontrado"}

# Rota para listar arquivos de uma pasta hash
@app.get("/audio/{hash_folder}")
async def list_audio_files(hash_folder: str):
    directory = f"/app/final_audio/{hash_folder}/"
    if not os.path.exists(directory):
        return {"error": "Pasta não encontrada"}
    
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
    
    return {"hash_folder": hash_folder, "files": files}

# Rota para listar transcrições de uma pasta hash
@app.get("/transcriptions/{hash_folder}")
async def list_transcription_files(hash_folder: str):
    directory = f"/app/transcriptions/{hash_folder}/"
    if not os.path.exists(directory):
        return {"error": "Pasta de transcrições não encontrada"}
    
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0)
    
    return {"hash_folder": hash_folder, "files": files}

# Rota para apagar toda uma pasta hash (áudio)
@app.delete("/audio/{hash_folder}")
async def delete_hash_folder(hash_folder: str):
    directory = f"/app/final_audio/{hash_folder}/"
    if os.path.exists(directory):
        shutil.rmtree(directory)
        return {"message": f"Pasta de áudio {hash_folder} foi apagada com sucesso."}
    else:
        return {"error": "Pasta não encontrada"}

# Rota para apagar toda uma pasta hash (transcrições)
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
    # Configurações básicas
    uvicorn.run(app, host="0.0.0.0", port=7010)
