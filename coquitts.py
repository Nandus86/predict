from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

# Define o modelo de entrada para o FastAPI
class DadosEntrada(BaseModel):
    texto_prompt: str
    idioma: str
    audio_referencia: str
    usar_microfone: str
    limpar_voz: bool
    nao_detectar_auto: bool
    concordo: bool
    indice_funcao: int
    usar_mic: bool

app = FastAPI()

# Cria o cliente Gradio uma vez, fora da função
cliente = Client("https://coquitts.nandus.com.br/")

@app.post("/prever")
async def prever(dados: DadosEntrada):
    try:
        # Chama o método predict da API Gradio passando os parâmetros diretamente
        resultado = cliente.predict(
            dados.texto_prompt,
            dados.idioma,
            dados.audio_referencia,
            dados.usar_microfone,
            dados.usar_mic,  # Nota: Alterado a ordem para corresponder ao exemplo anterior
            dados.limpar_voz,
            dados.nao_detectar_auto,
            dados.concordo,
            fn_index=dados.indice_funcao
        )
        return {"resultado": resultado}
    except ValueError as e:
        # Captura erros específicos relacionados a parâmetros
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Captura outros erros
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7010)
