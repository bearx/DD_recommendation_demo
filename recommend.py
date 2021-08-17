import requests
import numpy as np
import scipy.sparse.linalg as slinalg
import scipy.sparse
cookies=""#your cookies
acc=20
k=10
def cos_sim(x,y):
    nemerator = x * y.T
    denominator = np.sqrt(x * x.T) * np.sqrt(y * y.T)
    return (nemerator / denominator)[0,0]
ddurl="https://api.vtbs.moe/v1/info"
allvtb=requests.get(ddurl).json()
sallvtb=dict()
lallvtb=set()
for x in allvtb:
    sallvtb[x["mid"]]=f"https://live.bilibili.com/{x['roomid']}"
    if x["liveStatus"]!=0:
        lallvtb.add(x["mid"])
def chkalive(vid):
    if vid in lallvtb:
        return (True,sallvtb[vid])
    return (False,sallvtb[vid])
print("----------vtb fetched----------")
ddurl="https://api.vtbs.moe/v1/guard/all"
allinfo=requests.get(ddurl).json()
mp_usr,mp_vtb=dict(),dict()
rev_vtb=dict()
def mpusr(uid):
    if uid in mp_usr.keys():
        return mp_usr[uid]
    t=mp_usr[uid]=len(mp_usr)
    return t
def mpvtb(vid):
    if vid in mp_vtb.keys():
        return mp_vtb[vid]
    t=mp_vtb[vid]=len(mp_vtb)
    rev_vtb[t]=vid
    return t
print("----------dd mat fetched----------")
def fetch_bilibili(uid):
    burl="https://api.bilibili.com/x/relation/followings?vmid={}&pn={}&ps=50&order=asc&jsonp=jsonp"
    burl=burl.format(uid,"{}")
    curl="https://api.bilibili.com/x/relation/followings?vmid={}&pn={}&ps=50&order=desc&jsonp=jsonp"
    curl=curl.format(uid,"{}")
    dds=set()
    rg=6
    if cookies!="":
        turl=burl.format(1)
        tot=requests.get(turl,cookies=cookies).json()["data"]["total"]
        rg=tot/50+1
    for i in range(1,rg):
        turl=burl.format(i)
        sinfo=requests.get(turl,cookies=cookies).json()["data"]["list"]
        for y in sinfo:
            if y["mid"] in mp_vtb.keys():
                dds.add(mpvtb(y["mid"]))
    if rg!=6:
        for i in range(1,rg):
            turl=curl.format(i)
            sinfo=requests.get(turl,cookies=cookies).json()["data"]["list"]
            for y in sinfo:
                if y["mid"] in mp_vtb.keys():
                    dds.add(mpvtb(y["mid"]))
    return dds
print("----------creating dd matrix----------")
for uid,usrinfo in allinfo.items():
    mpusr(uid)
    du=usrinfo['dd']
    for gu in range(3):
        for x in du[gu]:
            mpvtb(x)
nu,nv=len(mp_usr),len(mp_vtb)
youruid=input("now,input your uid:")
stu=-1
if youruid in mp_usr.keys():
    stu=mpusr(youruid)
    print("你在DD列表里")
if stu==-1:
    stu=nu
    nu+=1
    print("DD列表没你")
mat=np.zeros(shape=(nu,nv))
for uid,usrinfo in allinfo.items():
    du=usrinfo['dd']
    mu=mpusr(uid)
    for gu in range(3):
        for x in du[gu]:
            mat[mu,mpvtb(x)]=3**(gu+1)
dtd=fetch_bilibili(youruid)
sdtd={rev_vtb[x] for x in dtd}
print(sdtd)
for x in dtd:
    if mat[stu,x]==0:
        mat[stu,x]=1
mat=np.mat(mat)
print("----------ready for recommend----------")
n=np.shape(mat)[1]
U,Sigma,VT=slinalg.svds(mat,k=acc)#np.linalg.svd(mat)
sig2=Sigma**2
cut=0
for i in range(n):
    if sum(sig2[:i])/sum(sig2)>0.9:
        cut=i
        break
Sig4=np.mat(np.eye(cut)*Sigma[:cut])
svdr=mat.T*U[:,:cut]*Sig4.I
#svd
mt=mat.T
m=np.shape(svdr)[0]
w=np.mat(np.zeros((m,m)))
for i in range(m):
    for j in range(i,m):
        if j!=i:
            w[i,j]=cos_sim(svdr[i,:],svdr[j,:])
            w[j,i]=w[i,j]
        else:
            w[i,j]=0
#recommend
m,n=np.shape(mt)
interaction=mt[:,stu].T
not_inter=[]
for i in range(m):
    if interaction[0,i]==0:
        not_inter.append(i)
predict={}
for x in not_inter:
    item=np.copy(interaction)
    for j in range(m):
        if item[0,j]!=0:
            if x not in predict:
                predict[x]=w[x,j]*item[0,j]
            else:
                predict[x]=predict[x]+w[x,j]*item[0,j]
res=sorted(predict.items(), key=lambda d:d[1], reverse=True)
print("----------all prediction----------")
top_recom=[]
len_result=len(res)
now,cur=0,0
while now<k and cur<len_result:
    if mat[stu,cur]==0:
        top_recom.append((chkalive(rev_vtb[res[cur][0]]),res[cur][1]))
        now+=1
    cur+=1
for id,x in enumerate(top_recom):
    print(id,':',str(x))
print("----------alive prediction----------")
top_recom=[]
len_result=len(res)
now,cur=0,0
while now<k and cur<len_result:
    if mat[stu,cur]==0 and chkalive(rev_vtb[res[cur][0]])[0]:
        top_recom.append((chkalive(rev_vtb[res[cur][0]]),res[cur][1]))
        now+=1
    cur+=1
for id,x in enumerate(top_recom):
    print(id,':',str(x))
