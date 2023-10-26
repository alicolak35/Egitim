import random
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, SimpleRNN, MaxPool2D,GRU,LSTM ,Flatten,Reshape, Input \
            ,SeparableConv2D,AvgPool2D,BatchNormalization,Conv2D, GlobalMaxPool2D, Dropout,InputLayer, DepthwiseConv2D
from keras.utils import to_categorical,plot_model
from keras.applications.mobilenet_v3 import MobileNetV3Small
from glob import glob
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib

matplotlib.use("Agg")

konum_icbhi_1000 = r"C:\Users\uniha\PycharmProjects\pythonProject1\Audio Files\icbhi ek\png\100-1000hz"
dosya_yolu_icbhi_1000 =glob(konum_icbhi_1000 + "/*.png")

konum_Lungset_1000 =r"C:\Users\uniha\PycharmProjects\pythonProject1\Audio Files\Diafram,filtreli,100-1000\PNG"
dosya_yolu_Lungset_1000 = glob(konum_Lungset_1000 + "/*.png")

csv_icbhi = r"C:\Users\uniha\PycharmProjects\pythonProject1\patient_diagnosis_25.02.csv"
csv_Lungset = r"C:\Users\uniha\PycharmProjects\pythonProject1\veri2_düzenleme.csv"

image_size = (40, 40)# Her bir saniyede 8 adet desibel değeri var. Bu boyut ile veri setimde en fazla sayıda bulunan 5 saniyelik verileri kayıpsız olarak çevrilecek
df_icbhi = pd.read_csv(csv_icbhi)
df_Lungset = pd.read_csv(csv_Lungset)

diagnosis_dict = {}
for index, row in df_icbhi.iterrows():
    patient_id = row["ID"]
    diagnosis = row["Diagnosis"]
    diagnosis_dict[patient_id] = diagnosis

Lungset_dict = {}
for index, row in df_Lungset.iterrows():
    patient_id = row["ID"]
    diagnosis = row["Diagnosis"]
    Lungset_dict[patient_id] = diagnosis

def load_image(file_path):
    image = Image.open(file_path)
    image_resized = image.resize((image_size[1], image_size[0]))
    rgb_image = image_resized.convert("RGB")
    image_array = np.asarray(rgb_image)
    return image_array

healthy = []
copd = []
asthma = []

healthy_label = []
asthma_label = []
copd_label = []

##aşağıdaki yorumlu olan sınıfları denediğim sınıflandırmaya göre değiştirdim.
for dosya_uzantisi in (dosya_yolu_Lungset_1000 ):
    
        dosya_adi = dosya_uzantisi.split("\\")[-1]
        patient_id = int(dosya_adi.split("_")[0])
        if patient_id in Lungset_dict:
                diagnosis = Lungset_dict[patient_id]
                if diagnosis == "COPD":
                    continue
                    #copd.append(dosya_uzantisi)
                    #label = 0
                    #copd_label.append(label)
                elif (diagnosis == "Asthma"):
                    label = 0
                    asthma.append(dosya_uzantisi)
                    asthma_label.append(label)
                elif diagnosis == "Healthy":
                    label = 1
                    healthy.append(dosya_uzantisi)
                    healthy_label.append(label)

for dosya_uzantisi in dosya_yolu_icbhi_1000:
        dosya_adi = dosya_uzantisi.split("\\")[-1]
        patient_id = int(dosya_adi.split("_")[0])
        if patient_id in diagnosis_dict:
                #file_path = join(data_dir, file_name)
                # image_paths.append(file_path)
                diagnosis = diagnosis_dict[patient_id]
                if diagnosis == "COPD":
                    continue
                    #copd.append(dosya_uzantisi)
                    #label = 0
                    #copd_label.append(label)
                elif diagnosis == "Asthma":
                    label = 0
                    asthma.append(dosya_uzantisi)
                    asthma_label.append(label)
                elif diagnosis == "Healthy":
                    label = 1
                    healthy.append(dosya_uzantisi)
                    healthy_label.append(label)

He_COPD = np.concatenate((np.array(healthy),np.array(copd)))
As_COPD = np.concatenate((np.array(asthma),np.array(copd)))
He_As = np.concatenate((np.array(healthy),np.array(asthma)))


He_COPD_label = np.concatenate((np.array(healthy_label),np.array(copd_label)))
Asthma_COPD_label = np.concatenate((np.array(asthma_label),np.array(copd_label)))
Asthma_He_label = np.concatenate((np.array(asthma_label),np.array(healthy_label)))

num_samples = len(He_As)
images = np.empty((num_samples, image_size[0] ,image_size[1], 1))#(40,40,1) boyutlarında öznitelik çıktı

for i, image_path in enumerate(He_As):
    image = load_image(image_path)
    images[i] = image

labels = np.array(Asthma_He_label)

indices = np.arange(num_samples)

np.random.shuffle(indices)

train_indices =indices[:int(0.8*len(labels))]
val_indices = indices[int(0.8*(num_samples)):]

scaler = MinMaxScaler(feature_range=(0,1))# desibel değerleri 0'dan küçük olduğu için -1,1 yerine 0,1 arasında normalize ettim.

train_images= images[train_indices]
print(train_images.shape)

orignal_shape= train_images.shape
train_images = np.reshape(train_images,(len(train_images[0]),-1)#Diziyi 1 boyutlu hale getirdi
scaler.fit(train_images)
train_images = scaler.transform(train_images)
train_images = np.reshape(train_images,orignal_shape)
train_labels = labels[train_indices]
b0=[]
b1=[]
b2=[]

for i in train_labels:
    if i == 1:
        b1.append(i)
    elif i == 2:
        b2.append(i)
    else:
        b0.append(i)

train_catlabels = to_categorical(train_labels,num_classes=2)


val_images = images[val_indices]
orignal_shape= val_images.shape
val_images = np.reshape(val_images,(len(val_images[0]),-1))
scaler.fit(val_images)
val_images = scaler.transform(val_images)
val_images = np.reshape(val_images,orignal_shape)

val_labels = labels[val_indices]

bos0= []
bos1= []
bos2= []


for i in val_labels:
    if i == 1:
        bos1.append(i)
    elif i == 2:
        bos2.append(i)
    else:
        bos0.append(i)

val_catlabels = to_categorical(val_labels,num_classes=2)

def zamanla_isimlendirme():
    isim = time.strftime('%c')
    isim = isim[11:19]
    isim = isim.replace(":", ".")
    return isim

def acc_loss_plottin(isim,history):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.suptitle("Accuracy and Loss of Validation and Training sets")
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.legend(["Training", "Validation"])

    plt.subplot(212)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.title("Model loss")
    plt.legend(["Training loss", "Validation Loss"])
    plt.savefig(f"{isim}_loss_acc.png")
    return

def report_Astım_KOAH(model,isim,batch):
    y_pred = model.predict(val_images, batch_size=batch)
    y_pred = np.argmax(y_pred, axis=1)
    val_label = val_labels.reshape(-1, 1)
    cm = confusion_matrix(val_label, y_pred)
    cr = classification_report(val_label, y_pred)

    print("Classification Report: \n", cr)

    plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="crest", xticklabels=["COPD","Asthma"], yticklabels=["COPD","Asthma"])
    figure = ax.get_figure()
    figure.savefig(f"{isim}_heatmap.png")
    return print("raporlama başarılı")

def report_KOAH_Sag(model,isim,batch):
    y_pred = model.predict(val_images, batch_size=batch)
    y_pred = np.argmax(y_pred, axis=1)
    val_label = val_labels.reshape(-1, 1)
    cm = confusion_matrix(val_label, y_pred)
    cr = classification_report(val_label, y_pred)

    print("Classification Report: \n", cr)

    plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="crest", xticklabels=["Asthma","Healthy"], yticklabels=["Asthma","Healthy"])
    figure = ax.get_figure()
    figure.savefig(f"{isim}_heatmap.png")
    return print("raporlama başarılı")

def report(model,isim,batch):
    y_pred = model.predict(val_images, batch_size=batch)
    y_pred = np.argmax(y_pred, axis=1)
    val_label = val_labels.reshape(-1, 1)
    cm = confusion_matrix(val_label, y_pred)
    cr = classification_report(val_label, y_pred)

    print("Classification Report: \n", cr)

    plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap="crest", xticklabels=["Asthma","COPD","Healthy"], yticklabels=["Asthma","COPD","Healthy"])
    #ax.set(xlabel="Prediction", ylabel="Actual")

    figure = ax.get_figure()
    figure.savefig(f"{isim}_heatmap.png")
    return print("raporlama başarılı")


def save_plot(model,isim):
     #plot_model(model, to_file=f"{isim}.png", show_shapes=True, show_layer_activations=True, dpi=124)
     model.save(f"{isim}.h5")
     return print("save_plot başarılı")


def simpleRNN(lr,batch):
    base_model = MobileNetV3Small(include_top=False, input_tensor=Input(shape=(image_size[0], image_size[1], 3)), pooling=None, weights="imagenet",alpha=1.0, include_preprocessing=False,minimalistic=True)
    base_model.trainable = False
    x = base_model.output
    x = GlobalMaxPool2D()
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=x)
    model.compile(metrics=["accuracy"],loss="categorical_crossentropy",optimizer=Adam(learning_rate=lr))
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, verbose=0)

    history = model.fit(train_images, train_catlabels, batch_size=batch, epochs=120,
                        validation_data=(val_images, val_catlabels), validation_batch_size=batch,
                        callbacks=[early_stopping])

    isim = zamanla_isimlendirme() +  f"4x576,lr={lr},batch{batch}_astım-Sağ_100-1000hz"
    save_plot(model, isim)
    report_KOAH_Sag(model,isim,batch)
    acc_loss_plottin(isim,history)
    return model

model = KerasClassifier(build_fn=simpleRNN, verbose=0)
#    "RNNtype": [LSTM,GRU,SimpleRNN],    "dr": [0.15,0.25],
#     "RNN":[LSTM,SimpleRNN],
#     "birim" : [11,8],
#     "aktivasyon" : ["relu","tanh"]
parameters = {
    "lr" : [0.0001],
     "batch":[6],

}

grid =GridSearchCV(estimator=model, param_grid=parameters, cv=9, scoring="accuracy",n_jobs=3)#accuracy doğrulama kümesindeki isbetliliğ itemsil diyoumş
grid_result = grid.fit(train_images, train_catlabels)

# En iyi sonuçları ekrana yazdırma
print("Best Score: ", grid_result.best_score_)
print("Best Parameters: ", grid_result.best_params_)
best_model = grid_result.best_estimator_


