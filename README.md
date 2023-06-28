# RVC-Easy-Infer
![RVC-Easy-Infer](https://cdn.discordapp.com/attachments/665444039104921665/1123499308566708285/image.png)

## Preparando el entorno
- Instalar Python 3.8 o superior si no lo tienes
- Ejecutar los siguientes comandos

Para tarjetas graficas de Nvidia
```bat
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
Otras
```bat
python -m pip install -U pip setuptools wheel
pip install -U torch torchaudio 
pip install -r requirements.txt
```

Ejecuta el archivo `start_gui.bat` para iniciar la aplicación.
Automaticamente descargará el archivo [hubbert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt) para poder hacer inferencia, o simplemente descarguelo y ponlo en la raiz del programa.

