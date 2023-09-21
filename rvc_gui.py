import os
import sys
import traceback, pdb
import warnings
import customtkinter as ctk
import tkinter
import zipfile
from PIL import Image, ImageTk
import numpy as np
import torch
import threading
import soundfile as sf
from assets.configs.config import Config
from fairseq import checkpoint_utils
from lib.infer.infer_libs.audio import load_audio
from lib.infer.modules.vc.modules import VC
from dotenv import load_dotenv
load_dotenv()
config = Config()
vc = VC(config)

from lib.infer.modules.vc.utils import *
import lib.globals.globals as rvc_globals
import wget
 
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(os.path.join(now_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "output"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "audios"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


hubert_model = None

if not os.path.exists(os.path.join(now_dir, "assets", "hubert", "hubert_base.pt")):
    print("Descargando modelo base...")
    try:
        wget.download(url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")
    except:
        print("Hubo un error descargando el modelo, intenta descargarlo manualmente y ponerlo dentro del folder del programa.")
        print("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt")


if not os.path.exists(os.path.join(now_dir, "assets", "rmvpe", "rmvpe.pt")):
    print("Descargando modelo rmvpe...")
    try:
        wget.download(url="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt")
    except:
        print("Hubo un error descargando el modelo, intenta descargarlo manualmente y ponerlo dentro del folder del programa.")
        print("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt")



class App(ctk.CTk):
    def __init__(self, models_dir = "./models", height = 600, width = 800):
        super().__init__()
        
        self.device = config.device
        self.is_half = config.is_half
        print(self.device, self.is_half)
        
        self._set_appearance_mode("System")
        #ctk.set_default_color_theme("")
        self.model_loaded = False
        
        self.logo = ctk.CTkImage(Image.open("assets/images/icon.png"), size=(40, 40))
        self.image_button = ctk.CTkButton(master=self, text="Applio Rvc Fork EASY GUI", image=self.logo)
        self.image_button.pack()
        
        self.master_frame = ctk.CTkFrame(master=self, height=self.winfo_height() - 20, width=self.winfo_width() - 20)
        # Inicializar la seccion de la salida de audio
        self.output_audio_frame = ctk.CTkFrame(master=self)
        # Inicializar el widget de estado
        self.result_state = ctk.CTkLabel(self, text="", height=50, width=400, corner_radius=10)

        self.container = ctk.CTkFrame(master=self.master_frame, width=250)
        
        self.inputpath_frame = ctk.CTkFrame(master=self.container)
        self.input_audio_label = ctk.CTkLabel(self.inputpath_frame, text="Selecciona tu acapella:")
        self.input_audio_entry = ctk.CTkEntry(self.inputpath_frame)
        self.browse_button = ctk.CTkButton(self.inputpath_frame, text="Buscar", command=self.browse_file, 
                                           fg_color="#ffffff", text_color="#018ada",
                                           hover_color="#ffffff"
                                           )
        
        self.select_model_frame = ctk.CTkFrame(self.container)
        
        self.select_model = ctk.StringVar(value="Seleccion un modelo")
        self.models_dir = models_dir

        #self.checkbox.place(x=40, y=70)
        
        self.model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(
            models_dir, f)) and any(f.endswith(".pth") for f in os.listdir(os.path.join(models_dir, f)))]
        
        print(self.model_folders)
        
        self.model_list = ctk.CTkOptionMenu(self.select_model_frame, values=self.model_folders,
                               command=self.selected_model,
                               variable=self.select_model, fg_color="#ffffff", text_color="#018ada"
        )
        
        self.file_index_entry = ctk.CTkEntry(self.container, width=250)       
        self.pitch_frame = ctk.CTkFrame(self.container)
        self.sid_label = ctk.CTkLabel(self.select_model_frame, text="ID del Hablante:")
        self.sid_entry = ctk.CTkEntry(self.select_model_frame)
        self.sid_entry.insert(0, "0")
        self.sid_entry.configure(state="disabled")
        
        self.buttons_container = ctk.CTkFrame(master=self.container)
        
        self.f0_method_label = ctk.CTkLabel(self.pitch_frame, text="Algoritmo")
        
        #["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny"]
        self.f0_method_entry = ctk.CTkSegmentedButton(
            self.pitch_frame, height=40, 
            values=["dio", "pm","harvest", "crepe", "crepe-tiny", "mangio-crepe" , "mangio-crepe-tiny", "rmvpe"], 
            command=self.crepe_hop_length_slider_visibility
        )
        self.f0_method_entry.set("dio")
        
        self.f0_pitch_label = ctk.CTkLabel(self.pitch_frame, text="Tono: 0")
        self.f0_pitch_entry = ctk.CTkSlider(self.pitch_frame, from_=-20, to=20, number_of_steps=100, command=self.pitch_slider_event, )
        self.f0_pitch_entry.set(0)
        
        self.crepe_hop_length_label = ctk.CTkLabel(self.pitch_frame, text="crepe hop: 128")
        self.crepe_hop_length_entry = ctk.CTkSlider(
            self.pitch_frame, from_=1, to=8, number_of_steps=7, command=self.crepe_hop_length_slider_event)
        self.crepe_hop_length_entry.set(2)
        
        # Iniciarlizar el widget index
        self.file_index_label = ctk.CTkLabel(self.pitch_frame, text="Archivo .index (Recomendado)")
        self.file_index_entry = ctk.CTkEntry(self.pitch_frame, width=250)
        
        # Inicializar el widget de index ratio
        self.index_rate_entry = ctk.CTkSlider(
            self.pitch_frame, from_=0, to=1, number_of_steps=20, command=self.index_slider_event, )
        self.index_rate_entry.set(0.4)
        self.index_rate_label = ctk.CTkLabel(self.pitch_frame, text="Tasa de recuperación de caracteristicas: 0.4")
        
        # intiilizing import models button widget
        self.import_moodels_button = ctk.CTkButton(
            self.select_model_frame, fg_color="#ffffff", 
            text_color="#018ada",
            hover_color="#ffffff", corner_radius=5, text="Importar modelo desde .zip", command=self.browse_zip)
        
        # intiilizing run button widget
        self.run_button = ctk.CTkButton(
            self.buttons_container, fg_color="green", hover_color="darkgreen", text="Convertir", command=self.start_processing)
                
        # intiilizing last output label & open output button widget
        self.last_output_label = ctk.CTkLabel(self.output_audio_frame, text="Ruta de salida: ")
        self.last_output_file = ctk.CTkLabel(self.output_audio_frame, text="", text_color="green", width=self.winfo_width())
        
        # intiilizing loading progress bar widget
        self.loading_frame = ctk.CTkFrame(master=self, width=200) 
        self.laoding_label = ctk.CTkLabel(self.loading_frame, text="Convirtiendo..., Si la ventana no responde, Por favor espera.")
        self.loading_progress = ctk.CTkProgressBar(master=self.loading_frame, width=200)
        self.loading_progress.configure(mode="indeterminate")
        
        self.last_output_file = ctk.CTkLabel(self.output_audio_frame, text="", text_color="green", width=self.winfo_width())
        
        # Inicializar cambiar dispositivo
        self.change_device_label = ctk.CTkLabel(self.select_model_frame, text="Modo de procesamiento")
        self.change_device = ctk.CTkSegmentedButton(
            self.select_model_frame, command=lambda value: self.update_config(value))
        self.change_device.configure(values=["GPU", "CPU"])
        
        if "cpu" in self.device.lower() or self.device.lower() == "cpu":
            self.change_device.set("CPU")
            self.change_device.configure(state="disabled")
        
        else:
            self.change_device.set("GPU")
            
        self.open_file_explorer_button = ctk.CTkButton(self.buttons_container, text="Abrir folder", command=self.open_file_explorer)
        
        # Mostrar contenedor principal
        self.master_frame.pack(padx=5, pady=5)
        # Mostrar contenedor izquierdo
        self.container.grid(row=0, column=0, padx=10,  pady=10, sticky="nsew")
        # Mostrar contenedor de ruta acapella
        self.inputpath_frame.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")
        # Mostrar label de acapella
        self.input_audio_label.grid(padx=10, pady=10, row=0, column=0)
        # Mostrar input de acapella
        self.input_audio_entry.grid(padx=10, pady=10, row=0, column=1)
        # Mostrar boton para agregar acapella
        self.browse_button.grid(padx=10, pady=10, row=0, column=2)
        
        # Mostrar el boton de importar modelo
        self.import_moodels_button.grid(row=0, column=0, padx=10, pady=10)   
        
        # Mostrar el contenedor de seleccion de modelo
        self.select_model_frame.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")
        # Mostrar lista de modelos
        self.model_list.grid(padx=10, pady=10, row=0, column=2)   
        
        # Mostrar frame de tono
        self.pitch_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        
        # Mostrar frame de botones
        self.buttons_container.grid(row=4, column=0, padx=15, pady=10, sticky="nsew")
        # Mostrar label de tono
        self.f0_method_label.grid(padx=10, pady=10, row=0, column=0)        
        # Mostrar algoritmos
        self.f0_method_entry.grid(padx=10, pady=10, row=0, column=1)
        # Mostrar label de semitonos
        self.f0_pitch_label.grid(padx=10, pady=10, row=3, column=0)
        
        self.file_index_label.grid(padx=10, pady=10, row=5, column=0)
        self.file_index_entry.grid(padx=10, pady=10, row=5, column=1)
        
        # Mostrar slider de semitonos
        self.f0_pitch_entry.grid(padx=10, pady=10, row=3, column=1)
        # Mostrar label de index ratio
        self.index_rate_label.grid(padx=10, pady=10, row=4, column=0)
        # Mostrar slider de index ratio
        self.index_rate_entry.grid(padx=10, pady=10, row=4, column=1)
        # Mostrar boton de convertir
        
        self.change_device_label.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        self.change_device.grid(row=1, column=2, columnspan=2, padx=10, pady=5)
        
        self.run_button.grid(padx=30, pady=30, row=0, column=0, columnspan=2)
        
        self.laoding_label.pack(padx=10, pady=10)
        self.loading_progress.pack(padx=10, pady=10)
        
        # Mostrar label input de salida
        self.last_output_label.grid( pady=10, row=0, column=0)
        # Mostrar input de salida
        self.last_output_file.grid( pady=10, row=0, column=1)
        
        self.open_file_explorer_button.grid(pady=10, padx=10, row=0, column=3)
        
        # self.button = ctk.CTkButton(master=self, text="Click", command=self.button_test)
        # self.button.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
    
    def update_config(self,selected):
        
        if selected == "GPU":
            self.device = "cuda:0"
        # is_half = True
        else:
            if torch.backends.mps.is_available():
                self.device  = "mps"
        #  is_half = False
            else: 
                self.device = "cpu"
                self.is_half = False

        config.device = self.device
        config.is_half = self.is_half
        

        if "pth_file_path" in globals():
            load_hubert()
            vc.get_vc(pth_file_path, 0)


    def extract_model_from_zip(self, zip_path, output_dir):
        # Extract the folder name from the zip file path
        folder_name = os.path.splitext(os.path.basename(zip_path))[0]

        # Create a folder with the same name as the zip file inside the output directory
        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if (member.endswith('.pth') and not (os.path.basename(member).startswith("G_") or os.path.basename(member).startswith("D_")) and zip_ref.getinfo(member).file_size < 200*(1024**2)) or (member.endswith('.index') and not (os.path.basename(member).startswith("trained"))):
                    # Extract the file to the output folder
                    zip_ref.extract(member, output_folder)

                    # Move the file to the top level of the output folder
                    file_path = os.path.join(output_folder, member)
                    new_path = os.path.join(output_folder, os.path.basename(file_path))
                    os.rename(file_path, new_path)

        print(f"Archivos del modelo extraidos en el folder: {output_folder}")
    
    def refresh_model_list(self):
        self.model_folders = [f for f in os.listdir(self.models_dir) if os.path.isdir(os.path.join(
        self.models_dir, f)) and any(f.endswith(".pth") for f in os.listdir(os.path.join(self.models_dir, f)))]
        
        self.model_list.configure(values=self.model_folders)
        self.model_list.update()
        
    def browse_zip(self):
        global zip_file
        zip_file = tkinter.filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select file",
            filetypes=(("archivos zip", "*.zip"), ("all files", "*.*")),
        )
        self.extract_model_from_zip(zip_file, self.models_dir)
        self.refresh_model_list()
        
    def browse_file(self):
        filepath = tkinter.filedialog.askopenfilename (
            filetypes=[("Archivos de audio", "*.wav;*.mp3")])
        filepath = os.path.normpath(filepath)  # Normalize file path
        self.input_audio_entry.delete(0, tkinter.END)
        self.input_audio_entry.insert(0, filepath)

    def selected_model(self, choice):
        self.file_index_entry.delete(0, ctk.END)
        model_dir = os.path.join(self.models_dir, choice)
        pth_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) 
                    and f.endswith(".pth") and not (f.startswith("G_") or f.startswith("D_"))
                    and os.path.getsize(os.path.join(model_dir, f)) < 200*(1024**2)]
        
        if pth_files:
            global pth_file_path
            pth_file_path = os.path.join(model_dir, pth_files[0])
            npy_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)) 
                        and f.endswith(".index")]
            if npy_files:
                npy_files_dir = [os.path.join(model_dir, f) for f in npy_files]
                if len(npy_files_dir) == 1:
                    index_file = npy_files_dir[0]
                    print(f".pth file directory: {pth_file_path}")
                    print(f".index file directory: {index_file}")
                    self.file_index_entry.insert(0, os.path.normpath(index_file))
                else:
                    print(f"Conjunto incompleto de archivos .index encontrados en {model_dir}")
            else:
                print(f"No se encontró un archivo .index en {model_dir}")
            vc.get_vc(pth_file_path, 0.33, 0.33)
            
            self.model_loaded = True
        else:
            print(f"No se encontraron .pth seleccionables {model_dir}")

    def pitch_slider_event(self, value):
        self.f0_pitch_label.configure(text='Tono: %s' % round(value))
        
    def crepe_hop_length_slider_event(self,value):
        self.crepe_hop_length_label.configure(text='crepe hop: %s' % round((value) * 64))
        
    # Oculta el slider de crepe hop length si no se selecciona crepe
    def crepe_hop_length_slider_visibility(self, value):
        if value in ["crepe","crepe-tiny","mangio-crepe","mangio-crepe-tiny", "rmvpe"]:
            self.crepe_hop_length_label.grid(row=2, column=0, padx=10, pady=5, )
            self.crepe_hop_length_entry.grid(row=2, column=1, padx=10, pady=5, )
        else:
            self.crepe_hop_length_label.grid_remove()
            self.crepe_hop_length_entry.grid_remove()
        
    def index_slider_event(self, value):
        self.index_rate_label.configure(text='Feature retrieval rate: %s' % round(value, 2))
    
    def get_output_path(self, file_path):
        if not os.path.exists(file_path):
            # change the file extension to .wav
            
            return file_path  # File path does not exist, return as is

        # Split file path into directory, base filename, and extension
        dir_name, file_name = os.path.split(file_path)
        file_name, file_ext = os.path.splitext(file_name)

        # Initialize index to 1
        index = 1

        # Increment index until a new file path is found
        while True:
            new_file_name = f"{file_name}_RVC_{index}{file_ext}"
            new_file_path = os.path.join(dir_name, new_file_name)
            if not os.path.exists(new_file_path):
                # change the file extension to .wav
                new_file_path = os.path.splitext(new_file_path)[0] + ".wav"
                return new_file_path  # Found new file path, return it
            index += 1
    
    def start_processing(self):
        t = threading.Thread(target=self.start_inference)
        t.start()
    
    def start_inference(self):
        self.output_audio_frame.pack_forget()
        self.result_state.pack_forget()
        self.run_button.configure(state="disabled")
        # Variables temps
        minpitch_slider = 50
        minpitch_txtbox = 50
        maxpitch_slider = 1100
        maxpitch_txtbox = 1100
        # Get values from user input widgets
        autotune = False
        sid = self.sid_entry.get()
        input_audio = self.input_audio_entry.get()
        f0_pitch = round(self.f0_pitch_entry.get())
        crepe_hop_length = round((self.crepe_hop_length_entry.get()) * 64)
        f0_file = None
        f0_method = self.f0_method_entry.get()
        file_index = self.file_index_entry.get()
        # file_big_npy = file_big_npy_entry.get()
        index_rate = round(self.index_rate_entry.get(),2)
        
        file_path = os.path.join(now_dir , "assets", "audios", os.path.basename(input_audio))
        new_file = self.get_output_path(file_path)
        output_file =  os.path.join(now_dir , "assets", "audio-outputs", os.path.basename(new_file))
        
        print("sid: ", sid, "input_audio: ", input_audio, "f0_pitch: ", f0_pitch, "f0_file: ", f0_file, "f0_method: ", f0_method,
            "file_index: ", file_index, "file_big_npy: ", "index_rate: ", index_rate, "output_file: ", output_file)
        # Call the vc_single function with the user input values
        if self.model_loaded == True and os.path.isfile(input_audio):
            try:
                self.loading_frame.pack(padx=10, pady=10)
                self.loading_progress.start()
                
                result, audio_opt = vc.vc_single(
                    sid = 0,  input_audio_path1 = input_audio, 
                    f0_up_key = f0_pitch, f0_file = None, f0_method = f0_method, file_index2 = file_index,
                    file_index = file_index, index_rate = index_rate, filter_radius=3 ,resample_sr=0, protect=0.33, rms_mix_rate =1, crepe_hop_length = crepe_hop_length, output_path = output_file,
                    f0_min= minpitch_slider,
                    note_min= minpitch_txtbox,
                    f0_max= maxpitch_slider,
                    note_max= maxpitch_txtbox,
                    f0_autotune= autotune)
                
                # output_label.configure(text=result + "\n saved at" + output_file)
                print(os.path.join(output_file))
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(output_file) 
                        
                    self.run_button.configure(state="enabled")
                    message = result
                    self.result_state.configure(text_color="green")
                    self.last_output_file.configure(text=output_file)
                    self.output_audio_frame.pack(padx=10, pady=10)
                else: 
                    message = result
                    self.result_state.configure(text_color="red")

            except Exception as e:
                print(e)
                message = "Voice conversion failed", e

        # Update the output label with the result
        # output_label.configure(text=result + "\n saved at" + output_file)

            self.run_button.configure(state="enabled")
        else:
            message = "Por favor selecciona un modelo y un acapella."
            self.run_button.configure(state="enabled")
            self.result_state.configure(text_color="red")

        self.loading_progress.stop()
        self.loading_frame.pack_forget()
        self.result_state.pack(padx=10, pady=10, side="top")
        self.result_state.configure(text=message)
        
    def open_file_explorer(self):
        """Opens the File Explorer in the specified folder."""
        
        os.startfile(os.path.join(now_dir , "assets" ,"audio-outputs"))
        
  #  print(value)
  
app = App()
app.title("RVC - Uso Local - Applio Rvc Fork")

width = 800
height = 800
screen_height = app.winfo_screenheight()
app.geometry(f"{int(width)}x{int(height)}")

app.mainloop()
