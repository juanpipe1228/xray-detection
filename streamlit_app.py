import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.confidence = 0.25

    def set_params(self, model, confidence):
        self.model = model
        self.confidence = confidence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.model:
            results = self.model(img_rgb, conf=self.confidence)
            if results:
                annotated_frame = results[0].plot()
                print(annotated_frame)
                return av.VideoFrame.from_ndarray(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Detección de objetos prohibidos")
st.title("Daniel Villegas - Juan Rincon")


options = ["YOLO10L", "YOLO10X"]
df = pd.DataFrame(
    [
        {"Clase": "Bat", "Clase id": 0},
        {"Clase": "Gun", "Clase id": 1},
        {"Clase": "Battery", "Clase id": 2},
        {"Clase": "Dart", "Clase id": 3},
        {"Clase": "Fireworks", "Clase id": 4},
        {"Clase": "Hammer", "Clase id": 5},
        {"Clase": "Knife", "Clase id": 6},
        {"Clase": "Lighter", "Clase id": 7},
        {"Clase": "Pliers", "Clase id": 8},
        {"Clase": "Pressure vessel", "Clase id": 9},
        {"Clase": "Razor_blade", "Clase id": 10},
        {"Clase": "Saw_blade", "Clase id": 11},
        {"Clase": "Scissors", "Clase id": 12},
        {"Clase": "Screwdriver", "Clase id": 13},
        {"Clase": "Wrench", "Clase id": 14},

    ]
)
st.sidebar.title('Diccionario de clases')
st.sidebar.dataframe(df, hide_index=True)
# st.text('Bat: 0, Gun: 1, Battery: 2, Dart: 3, Fireworks: 4, Hammer: 5, Knife: 6, Lighter: 7, Pliers: 8, Pressure_vessel: 9, Razor_blade: 10, Saw_blade: 11, Scissors: 12, Screwdriver: 13, Wrench: 14')
choice = st.selectbox("Selecciona una modelo: ", options)

if choice == 'YOLO10L':
  model_path = 'best_v10l.pt'
  model = YOLO(model_path)
  print(model_path)

elif choice == 'YOLO10X':
  model_path = 'best_v10x.pt'
  model = YOLO(model_path)
  print(model_path)

camara = st.toggle("Camara en directo")

if camara == True:

  video_transformer = VideoTransformer()
  video_transformer.set_params(model, 0.60)
  webrtc_streamer(key="example", video_transformer_factory=lambda: video_transformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


elif camara == False:

  #Subir imagen
  image = st.file_uploader('Sube la imagen', type = ['png','jpg'])

  if image:
      image = Image.open(image)
      
      results = model(image, stream=False, conf=0.20)
      print('SALIDA:', results[0].boxes.cls) 
      if len(results) > 0:
        result = results[0]

        if len(result.boxes.cls) == 0:
          st.write('No se ha identificado ningun objeto')
          st.image(image=image)
        else:
          for box in result.boxes:
            print(box.cls.cpu().numpy())
            print(box.conf.cpu().numpy())
            
          # Save results to disk
          result.save(filename="result.jpg")

          result = Image.open('result.jpg')
          st.image(image=result)

## streamlit run app.py