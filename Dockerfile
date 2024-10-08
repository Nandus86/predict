# Usando imagem Python oficial
FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Criar diretório para arquivos de áudio final
RUN mkdir /app/final_audio

# Copiar o restante da aplicação
COPY coquitts.py .

# Expor a porta 7010
EXPOSE 7010

# Comando para rodar o FastAPI
CMD ["uvicorn", "coquitts:app", "--host", "0.0.0.0", "--port", "7010"]
