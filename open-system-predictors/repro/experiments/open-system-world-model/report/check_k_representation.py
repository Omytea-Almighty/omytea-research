"""Construct an exact rank-4 K predictor of the observable T1-simple process (technical report §3.2, Representability).
Post hoc check written during internal review round 6; not a fitted or registered result.
"""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from osw import tasks,models
torch.set_num_threads(1)
out={}
for tau in [20,80]:
    task=tasks.t1_simple(tau);lam=np.exp(-1/tau)
    T=tasks.cycle(8).T
    w=np.ones(4)*(1-lam)/4;w[1]+=lam
    Ks=np.stack([np.sqrt(w[j])*np.linalg.matrix_power(T,j) for j in range(4)])
    Ms=np.stack([np.diag(np.sqrt(task.B[:,o])) for o in range(4)])
    obs,act=tasks.sample(task,64,200,np.random.default_rng(616+tau));q=tasks.bayes_predictive(task,obs,act,200)
    # Double-precision explicit channel/filter, independent of Torch implementation.
    rho=np.broadcast_to(np.eye(8)/8,(len(obs),8,8)).copy()
    def channel(r):return np.einsum('rij,bjk,rlk->bil',Ks,r,Ks)
    for t in range(32):
        if t:rho=channel(rho)
        M=Ms[obs[:,t]];rho=M@rho@M.transpose(0,2,1);rho/=np.trace(rho,axis1=1,axis2=2)[:,None,None]
    pred=[]
    for k in range(200):
        rho=channel(rho);pred.append(np.einsum('oij,bji->bo',Ms.transpose(0,2,1)@Ms,rho))
    pred=np.stack(pred,axis=1)
    # Load the same construction into the actual registered rank-4 K class.
    m=models.build('K',4,1)
    with torch.no_grad():
        m.instrument.re.copy_(torch.tensor(Ms,dtype=torch.float32));m.instrument.im.zero_()
        m.g_re.copy_(torch.tensor(Ks.reshape(1,32,8),dtype=torch.float32));m.g_im.zero_()
        p=m(torch.tensor(obs),torch.tensor(act),200).numpy()
    out[str(tau)]={'kraus_rank':4,'trace_preservation_max_error':float(np.abs(np.einsum('rji,rjk->ik',Ks,Ks)-np.eye(8)).max()),'double_precision_max_prediction_error':float(np.abs(pred-q).max()),'registered_torch_class_max_prediction_error':float(np.abs(p-q).max()),'test_contexts':64,'horizons':200}
(Path(__file__).resolve().parent/'k-representability.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
