import os, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.benchmark=True
device="cuda" if torch.cuda.is_available() else "cpu"

dir_train="train"
dir_test="test"
csv_train="train.csv"
csv_test="test.csv"
out_csv="submisie_finala.csv"

DIM=224
NR_FOLDURI=5
EPOCI=90
BATCH=16
LR=1e-3
WD=2e-2
a_mixup=0.2
p_mixup=0.35
SEED=67

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
    if cu_label==True:
        y=np.array([int(v) for v in df['label']], dtype=np.int64)
        return X, y, nume
    return X, nume

def shift_img(img, max_pix=8):
    dy=np.random.randint(-max_pix, max_pix+1)
    dx=np.random.randint(-max_pix, max_pix+1)
    rez=np.roll(img, (dy, dx), axis=(0, 1))
    if dy>0: rez[:dy, :]=0
    elif dy<0: rez[dy:, :]=0
    if dx>0: rez[:, :dx]=0
    elif dx<0: rez[:, dx:]=0
    return rez

def mask_img(img, mm=20):
    r=img.copy()
    h, w=r.shape[0], r.shape[1]
    if np.random.rand()<0.4:
        y=np.random.randint(0, h-mm)
        r[y:y+mm, :]=0
    if np.random.rand()<0.4:
        x=np.random.randint(0, w-mm)
        r[:, x:x+mm]=0
    return r

class SetImagini(Dataset):
    def __init__(self, X, y=None, aug=False):
        self.X=X
        self.y=y
        self.aug=aug
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        img=self.X[i]
        if self.aug==True:
            if np.random.rand()<0.5: img=img[:, ::-1]
            if np.random.rand()<0.3: img=img[::-1, :]
            if np.random.rand()<0.5: img=shift_img(img, 8)
            if np.random.rand()<0.4: img=mask_img(img)
            img=np.ascontiguousarray(img)
        t=torch.from_numpy(img).unsqueeze(0)
        if self.y is None: return t
        return t, int(self.y[i])

def drop_path(x, p=0., train=False):
    if p==0. or train==False: return x
    keep=1-p
    shape=(x.shape[0], ) + (1, )*(x.ndim-1)
    m=(keep+torch.rand(shape, dtype=x.dtype, device=x.device)).floor_()
    return x.div(keep)*m

class ConvBNReLU(nn.Module):
    def __init__(self, ci, co, k=3, s=1, g=1):
        super().__init__()
        pad=(k[0]//2, k[1]//2) if isinstance(k, tuple) else k//2
        self.conv=nn.Conv2d(ci, co, k, s, pad, groups=g, bias=False)
        self.bn=nn.BatchNorm2d(co)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class CoordSpat(nn.Module):
    def __init__(self, c, red=4):
        super().__init__()
        self.ph=nn.AdaptiveAvgPool2d((None, 1))
        self.pw=nn.AdaptiveAvgPool2d((1, None))
        mid=max(c//red, 8)
        self.c1=nn.Conv2d(c, mid, 1, bias=False)
        self.bn=nn.BatchNorm2d(mid)
        self.ch=nn.Conv2d(mid, c, 1, bias=False)
        self.cw=nn.Conv2d(mid, c, 1, bias=False)
    def forward(self, x):
        h, w=x.shape[2], x.shape[3]
        xh=self.ph(x)
        xw=self.pw(x).transpose(2, 3)
        y=F.relu(self.bn(self.c1(torch.cat([xh, xw], dim=2))), inplace=True)
        xh, xw=torch.split(y, [h, w], dim=2)
        return x*torch.sigmoid(self.ch(xh))*torch.sigmoid(self.cw(xw.transpose(2, 3)))

class BlocProcesare(nn.Module):
    def __init__(self, ci, co, s=1, dp=0., coord=False):
        super().__init__()
        self.dp=dp
        self.skip=(s==1 and ci==co)
        self.coord=coord
        hid=ci*4
        self.expand=ConvBNReLU(ci, hid, k=1) if ci!=hid else nn.Identity()
        self.conv_3x3=ConvBNReLU(hid, hid, k=3, s=s, g=hid)
        self.conv_1x7=ConvBNReLU(hid, hid, k=(1, 7), s=s, g=hid)
        self.conv_7x1=ConvBNReLU(hid, hid, k=(7, 1), s=s, g=hid)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(hid, max(hid//4, 32), bias=False)
        self.relu1=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(max(hid//4, 32), hid*3, bias=False)
        if coord: self.ca=CoordSpat(hid)
        self.proj_conv=nn.Conv2d(hid, co, 1, bias=False)
        self.proj_bn=nn.BatchNorm2d(co)
    def forward(self, x):
        out=self.expand(x)
        u1=self.conv_3x3(out)
        u2=self.conv_1x7(out)
        u3=self.conv_7x1(out)
        u=u1+u2+u3
        s_tensor=self.gap(u).view(u.size(0), -1)
        atentie=F.softmax(self.fc2(self.relu1(self.fc1(s_tensor))).view(s_tensor.size(0), 3, u.size(1)), dim=1)
        v=u1*atentie[:, 0, :, None, None]+u2*atentie[:, 1, :, None, None]+u3*atentie[:, 2, :, None, None]
        if self.coord: v=self.ca(v)
        out=self.proj_bn(self.proj_conv(v))
        if self.skip: out=drop_path(out, self.dp, self.training)+x
        return out

class ModelProiect(nn.Module):
    def __init__(self, nr_cls, dpr=0.3):
        super().__init__()
        self.stem=ConvBNReLU(1, 48, s=2)
        self.blocks=nn.ModuleList([
            BlocProcesare(48, 48), BlocProcesare(48, 96, s=2), BlocProcesare(96, 96),
            BlocProcesare(96, 192, s=2), BlocProcesare(192, 192), BlocProcesare(192, 192, coord=True),
            BlocProcesare(192, 384, s=2, coord=True), BlocProcesare(384, 384, coord=True), BlocProcesare(384, 384, coord=True)
        ])
        rate_list=torch.linspace(0, dpr, 9).tolist()
        for i in range(len(self.blocks)): self.blocks[i].dp=rate_list[i]
        self.head=ConvBNReLU(384, 768, k=1)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.drop=nn.Dropout(0.4)
        self.fc=nn.Linear(768, nr_cls)
    def forward(self, x):
        x=self.stem(x)
        for b in self.blocks: x=b(x)
        return self.fc(self.drop(self.gap(self.head(x)).flatten(1)))

def mixup(x, y, alpha=a_mixup):
    lam=max(np.random.beta(alpha, alpha), 1-np.random.beta(alpha, alpha))
    perm=torch.randperm(x.size(0), device=x.device)
    return lam*x+(1-lam)*x[perm], y, y[perm], lam

def evaluare_model(model, X, y, nr_cls, bs=64):
    model.eval()
    s=np.zeros((len(X), nr_cls), dtype=np.float32)
    var=[X, X[:, :, ::-1].copy(), X[:, ::-1, :].copy(),
         np.array([shift_img(im, 4) for im in X]),
         np.array([shift_img(im[:, ::-1].copy(), 4) for im in X])]
    for Xv in var:
        off=0
        for b in DataLoader(SetImagini(Xv, y), batch_size=bs, shuffle=False):
            xb=b[0] if isinstance(b, (tuple, list)) else b
            with torch.no_grad():
                p_numpy=F.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            for j in range(len(p_numpy)):
                for k in range(nr_cls):
                    s[off+j][k]=s[off+j][k]+p_numpy[j][k]
            off+=len(p_numpy)
    return s/len(var)

def antrenare_fold(Xtr, ytr, Xval, yval, nr_cls, ponderi_clase, seed, nf):
    print("\n--- Fold "+str(nf)+"/"+str(NR_FOLDURI)+" (seed="+str(seed)+") ---")
    torch.manual_seed(seed); np.random.seed(seed)
    dl=DataLoader(SetImagini(Xtr, ytr, aug=True), batch_size=BATCH, shuffle=True)
    model=ModelProiect(nr_cls).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(dl), epochs=EPOCI)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(ponderi_clase, dtype=torch.float32).to(device), label_smoothing=0.05)

    best_acc=0.; model_salvat=None; best_pred=None; istoric_acc=[]; epoci_bune=[]
    for ep in range(EPOCI):
        t0=time.time(); model.train(); loss_total=0.
        for x, yb in dl:
            x, yb=x.to(device), yb.to(device)
            opt.zero_grad()
            if np.random.rand()<p_mixup:
                xm, ya, yc, lam=mixup(x, yb)
                loss=lam*crit(model(xm), ya)+(1-lam)*crit(model(xm), yc)
            else:
                loss=crit(model(x), yb)
            loss.backward(); opt.step(); sched.step()
            loss_total+=loss.item()*x.size(0)

        if ep==0 or (ep+1)%5==0 or ep>EPOCI-15:
            predictii=evaluare_model(model, Xval, yval, nr_cls).argmax(1)
            corecte=sum([1 for i in range(len(predictii)) if predictii[i]==yval[i]])
            acc_curenta=corecte/len(predictii)
            istoric_acc.append(acc_curenta); epoci_bune.append(ep+1)
            
            if acc_curenta>best_acc:
                best_acc=acc_curenta; best_pred=predictii
                model_salvat={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print("  ep "+str(ep+1)+"/"+str(EPOCI)+"  loss="+str(round(loss_total/len(Xtr), 4))+"  val="+str(round(acc_curenta, 4))+"  *best*")
            elif (ep+1)%5==0:
                print("  ep "+str(ep+1)+"/"+str(EPOCI)+"  loss="+str(round(loss_total/len(Xtr), 4))+"  val="+str(round(acc_curenta, 4)))

    print("  Scor fold "+str(nf)+": "+str(round(best_acc, 4)))
    model.load_state_dict(model_salvat)
    torch.save(model_salvat, "model_fold"+str(nf)+".pt")

    plt.figure(figsize=(8, 4)); plt.plot(epoci_bune, istoric_acc, 'r'); plt.title("Acuratete Fold "+str(nf))
    plt.savefig("acc_fold"+str(nf)+".png"); plt.close()
    plt.figure(figsize=(6, 5)); sns.heatmap(confusion_matrix(yval, best_pred), annot=True, fmt='g', cmap="Reds")
    plt.savefig("confuzie_fold"+str(nf)+".png"); plt.close()
    return model_salvat, best_acc

def main():
    print("incarc datele...")
    X, y_raw, _=citeste_set(csv_train, dir_train, True)
    Xtest, ids=citeste_set(csv_test, dir_test, False)

    labels=sorted(np.unique(y_raw).tolist())
    d={labels[i]: i for i in range(len(labels))}
    y=np.array([d[v] for v in y_raw], dtype=np.int64)

    ponderi_clase=[0.8, 1.0, 1.0, 1.4, 1.2]
    skf=StratifiedKFold(n_splits=NR_FOLDURI, shuffle=True, random_state=SEED)
    lista_stari=[]; lista_acurateti=[]; fold=0
    
    for itr, ival in skf.split(X, y):
        fold+=1
        st, acc=antrenare_fold(X[itr], y[itr], X[ival], y[ival], len(labels), ponderi_clase, SEED+fold, fold)
        lista_stari.append(st); lista_acurateti.append(acc)

    for i in range(len(lista_acurateti)): print("fold "+str(i+1)+": "+str(round(lista_acurateti[i], 4)))
    print("media: "+str(round(sum(lista_acurateti)/len(lista_acurateti), 4)))

    probe=np.zeros((len(Xtest), len(labels)), dtype=np.float32)
    for st in lista_stari:
        m=ModelProiect(len(labels)).to(device)
        m.load_state_dict(st)
        predictii_curente=evaluare_model(m, Xtest, None, len(labels))
        for i in range(len(probe)):
            for j in range(len(labels)): probe[i][j]+=predictii_curente[i][j]

    probe=probe/len(lista_stari)
    d_inv={i: labels[i] for i in range(len(labels))}
    pred_finale=np.array([d_inv[p] for p in probe.argmax(1)], dtype=np.int64)

    pd.DataFrame({"id": ids, "label": pred_finale}).to_csv(out_csv, index=False)
    print("gata, "+out_csv)

if __name__=="__main__":
    main()