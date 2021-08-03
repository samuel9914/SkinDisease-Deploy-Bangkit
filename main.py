import json
import joblib
import numpy as np
import os
import base64
from google.cloud import storage
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from google.cloud import storage
import time


from flask import Flask, request

app = Flask(__name__)



@app.route("/")
def hello():
    return "Hello, World!"

model = None
 # Model Bucket details
BUCKET_NAME        = "model-pkl-v1"
PROJECT_ID         = "capstone-team-b21-cap0080"

def download_model_file():

    keras_metadata =  "Model2-20210527T145508Z-001/Model2/keras_metadata.pb"
    saved_model = "Model2-20210527T145508Z-001/Model2/saved_model.pb"
    variables_data = "Model2-20210527T145508Z-001/Model2/variables/variables.data-00000-of-00001"
    variables_index = "Model2-20210527T145508Z-001/Model2/variables/variables.index"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob_keras_metadata     = bucket.blob(keras_metadata)
    blob_saved_model        = bucket.blob(saved_model)
    blob_variables_data     = bucket.blob(variables_data)
    blob_variables_index    = bucket.blob(variables_index)

    folder_model = '/tmp/Model2/'
    folder_model_var= '/tmp/Model2/variables/'
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)
        os.makedirs(folder_model_var)
    # Download the file to a destination
    blob_keras_metadata.download_to_filename(folder_model + "keras_metadata.pb")
   
    blob_variables_data.download_to_filename(folder_model_var + "variables.data-00000-of-00001")
    blob_variables_index.download_to_filename(folder_model_var + "variables.index")
    blob_saved_model.download_to_filename(folder_model + "saved_model.pb")
    

def get_image(path):
    #img=Image.open(path)
    #img=img.resize(224,224)
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x





@app.route("/predict", methods=["POST","GET"])
def predict():
    if request.method == "GET":
        time.sleep(0.2)
       	return ("dont use GET method/ change url to https")
    labels={2: {"disease":'Eczema(Peradangan kulit gatal)',
                "description":"Dermatitis atopik biasanya berkembang pada anak usia dini dan lebih sering terjadi pada orang yang memiliki riwayat keluarga dengan kondisi tersebut.Gejala utamanya adalah ruam yang biasanya muncul di lengan dan di belakang lutut, tetapi bisa juga muncul di mana saja.",
                "suggestion" : "Perawatan tergantung pada tingkat keparahan. Perawatan termasuk menghindari sabun dan bahan iritan lainnya. Krim atau salep tertentu juga dapat meredakan gatal. Beberapa saran perawatan untuk penyakit kulit ini seperti terapi sinar ultraviolet, menggunakan krim pereda gatal,terapi PUVA"
                },
            6: {"disease":'Scabies(Kudis)',
                "description":"Kondisi kulit yang sangat gatal dan menular yang disebabkan oleh tungau kecil yang bersembunyi. Kudis menular dan menyebar dengan cepat melalui kontak fisik yang dekat dalam keluarga, sekolah, atau panti jompo. Gejala kudis yang paling umum adalah rasa gatal yang hebat di area tempat tungau bersembunyi.",
                "suggestion": "Pengobatan terdiri dari anti parasit. Kudis dapat diobati dengan membunuh tungau dan telurnya dengan obat yang dioleskan dari leher ke bawah dan dibiarkan selama delapan jam. Tungau juga bisa dibunuh dengan obat oral."
                },
                
            1: {"disease":'chickenpox(Cacar air)',
                "description":"Infeksi virus yang sangat menular yang menyebabkan ruam gatal seperti lepuh pada kulit. Cacar air sangat menular kepada mereka yang belum pernah menderita penyakit ini atau telah divaksinasi.Gejala yang paling khas adalah ruam yang gatal seperti lepuh pada kulit.Cacar air dapat dicegah dengan vaksin. Perawatan biasanya melibatkan meredakan gejala, meskipun kelompok berisiko tinggi dapat menerima pengobatan antivirus.",
                "suggestion": "Pengobatan terdiri dari obat pereda nyeri. Cacar air dapat dicegah dengan vaksin. Perawatan biasanya melibatkan meredakan gejala, meskipun kelompok berisiko tinggi dapat menerima pengobatan antivirus. Selain itu perawatan dapat dilekukan dengan mencampurkan oatmeal dan air, dan juga bisa menggunakan pelembab."
                },
            0:{"disease": "Acne(Jerawat)",
               "description":"Kondisi kulit yang terjadi ketika folikel rambut tersumbat oleh minyak dan sel kulit mati.Jerawat paling sering terjadi pada remaja dan dewasa muda.",
               "suggestion":"Perawatan terdiri dari perawatan kulit.Perawatan termasuk krim dan pembersih yang dijual bebas, serta antibiotik resep.Biasanya digunakan antibiotik untuk menghentikan pertumbuhan atau membunuh bakteri."

               },
            3:{"disease":"Measles(Campak)",
               "description":"Infeksi virus yang serius untuk anak kecil tetapi mudah dicegah dengan vaksin.Penyakit ini menyebar melalui udara melalui tetesan pernapasan yang dihasilkan dari batuk atau bersin.",
               "suggestion":"Perawatan terdiri dari tindakan pencegahan. Tidak ada pengobatan untuk menghilangkan infeksi campak, tetapi obat penurun demam atau vitamin A yang dijual bebas dapat membantu mengatasi gejalanya."

               },
            4:{"disease":"Psoriasis",
               "desription": "Suatu kondisi di mana sel-sel kulit menumpuk dan membentuk sisik dan bercak-bercak yang gatal dan kering.Psoriasis dianggap sebagai masalah sistem kekebalan tubuh. Pemicunya termasuk infeksi, stres, dan pilek.",
               "suggestion":"Perawatan yang bertujuan untuk menghilangkan sisik dan menghentikan sel-sel kulit agar tidak tumbuh begitu cepat. Salep topikal, terapi cahaya, dan obat-obatan dapat memberikan kelegaan."
                },
            5:{"disease":"Ringworm(Kurap)",
               "description":"Infeksi jamur yang sangat menular pada kulit atau kulit kepala. Kurap menyebar melalui kontak kulit-ke-kulit atau dengan menyentuh hewan atau benda yang terinfeksi.",
               "suggestion":"Perawatan terdiri dari perawatan diri dan antijamur"
               }
                
            }





    
    # Use the global model variable 
    global model 
    start_time = time.time()
    if not model:
        print("NO MODEL LOADED YET")
        if not os.path.isfile("/tmp/Model2/variables/variables.data-00000-of-00001"):
            download_model_file()
        print("--Durasi download model-- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        if os.path.isfile("/tmp/Model2/variables/variables.data-00000-of-00001"):
            if  os.path.isfile("/tmp/Model2/saved_model.pb"):
        #print ("variables & pd exist")
                model = tensorflow.keras.models.load_model("/tmp/Model2")
            else:
                return ("no saved_model.pb")
        else:
            return ("no variables.data")
        #model = tensorflow.keras.models.load_model("/tmp/Model2", options=load_options)
        print("--Durasi load model-- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
    # Get the features sent for prediction

    

        
    request_json = request.json
    #print("data: {}".format(request_json))
    #print("type: {}".format(type(request_json)))

    if (request_json.get('features') is not None) :
        print("features exist in request")
        imgdata = base64.b64decode(request_json['features'].encode('utf-8'))
        filename = '/tmp/input_image.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
            f.close()
        client   = storage.Client(PROJECT_ID)
        if  os.path.isfile("/tmp/input_image.jpg"):
      #print("input image exist")
      #print(os.stat("/tmp/input_image.jpg").st_size)
            img, x = get_image(r'/tmp/input_image.jpg')
        else:
            return ("can't find input_image")
            print("--Durasi preprocess gambar-- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
        probabilities = model.predict([x])
        temp =0
        for i in range (len(probabilities[0])):
            temp = (i if probabilities[0][i]>probabilities[0][temp] else temp)
          #print(probabilities[0][i]>temp)
        result = {}
        result["prob"] = "{:.2f}%".format(100*probabilities[0][temp])
        result["disease"] = labels[temp]["disease"]
        result["description"] = labels[temp]["description"]
        result["suggestion"] = labels[temp]["suggestion"]
        
        print("--Durasi predict-- %s seconds ---" % (time.time() - start_time))
        return json.dumps(result), 200, {'Content-Type': 'application/json'}
    else:
        return "nothing sent for prediction"
 

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
