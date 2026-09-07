import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

dir_train="train"
dir_test="test"
csv_train="train.csv"
csv_test="test.csv"
out_csv="submisie_knn.csv"

DIM=64

def cit_img(cale):
    im=Image.open(cale).convert('L').resize((DIM, DIM), Image.BILINEAR)
    img=np.asarray(im, dtype=np.float32)
    return (img-img.mean())/(img.std()+1e-8)

def citeste_set(csv, folder, cu_label=True):
    df=pd.read_csv(csv)
    nume=df['id'].tolist()
    lista_x=[]
    for nm in nume:
        lista_x.append(cit_img(os.path.join(folder, nm)))
    X=np.array(lista_x, dtype=np.float32)
    
    # K-NN necesita vectori 1D, deci aplatizam imaginea 64x64 in 4096 features
    X=X.reshape(X.shape[0], -1)
    
    if cu_label==True:
        y=np.array([int(v) for v in df['label']], dtype=np.int64)
        return X, y, nume
    return X, nume

def main():
    print("incarc datele...")
    Xtrain, ytrain, _=citeste_set(csv_train, dir_train, True)
    Xtest, ids=citeste_set(csv_test, dir_test, False)

    print("antrenez KNN...")
    # k=5 este standardul de pornire; metrica e distanta Euclidiana
    knn=KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
    knn.fit(Xtrain, ytrain)

    print("generez predictii...")
    predictii=knn.predict(Xtest)

    pd.DataFrame({"id": ids, "label": predictii}).to_csv(out_csv, index=False)
    print("gata, "+out_csv)

if __name__=="__main__":
    main()