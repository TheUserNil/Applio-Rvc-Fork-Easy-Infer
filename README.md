# RVC-Easy-Infer
![Applio-Rvc-Fork-Easy-Infer](https://cdn.discordapp.com/attachments/989772841891270658/1154225605970239569/image.png)


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
Automaticamente descargará los archivos hubert_base.pt y rmvpe.pt para poder hacer inferencia, o simplemente descarguelo y ponlo en la raiz del programa.

