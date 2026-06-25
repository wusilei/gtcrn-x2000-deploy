#!/usr/bin/env python3
"""Complete MATLAB GTCRN fixed-point reference — all modules, self-contained."""
import numpy as np
from scipy.io import loadmat
import os

MAT_DIR = "/media/sf_haidesi/haidesi/gtcrn-x2000-deploy/GTCRN_speech_enhance_FPversion/para_in_mat"

_cache = {}
def W(name):
    if name not in _cache:
        d = loadmat(os.path.join(MAT_DIR, name+'.mat'))
        for k,v in d.items():
            if not k.startswith('__'): _cache[name] = np.asarray(v, np.float64); break
    return _cache[name].copy()

def FP(x, q):
    m = {'s32f20':(1,32,20),'s32f18':(1,32,18),'s16f15':(1,16,15),'s16f14':(1,16,14),
         's16f13':(1,16,13),'s16f12':(1,16,12),'s16f10':(1,16,10),
         'u32f20':(0,32,20),'u32f18':(0,32,18),'u16f15':(0,16,15),'u16f14':(0,16,14),
         'u16f13':(0,16,13),'u16f12':(0,16,12),'u16f10':(0,16,10)}[q]
    s,b,f=m; sc=2**f; mx=(1<<(b-1))-1 if s else (1<<b)-1; mn=-(1<<(b-1)) if s else 0
    xq=np.clip(np.round(np.float64(x)*sc),mn,mx).astype(np.int64)
    return xq, (xq/sc).astype(np.float32)

# ═══ Primitives ═══
def conv2d(x,w,b,Ci,Co,Ho,Wo,ks,st,pad,Qr):
    y=np.zeros((Co,Wo),np.int64)
    for oc in range(Co):
        yc=np.zeros(Wo,np.int64)
        for ic in range(Ci):
            xc=np.pad(x[ic],(pad[1],pad[1])) if pad[1]>0 else x[ic]
            for wi in range(Wo):
                ws=wi*st[1]; xk=xc[ws:ws+ks[1]]
                wk=w[oc,ic].ravel() if w.ndim==4 else w[oc].ravel()
                yc[wi]+=np.sum(np.round(xk.astype(np.float64)*wk*(2**Qr)).astype(np.int64))
        y[oc]=yc+b[oc]
    return y


def ptconv2d(x,w,b,Ci,Co,Ho,Wo,Qr):
    """Pointwise transposed conv: weight is (Cin,Cout,1,1) format"""
    wT=w.transpose(1,0,2,3) if w.ndim==4 else w.T  # → (Cout,Cin)
    return conv2d(x,wT,b,Ci,Co,Ho,Wo,[1,1],[1,1],[0,0],Qr)


def tconv2d(x,w,b,Ci,Co,Ho,Wo,ks,st,pad,Qr):
    """MATLAB tconv2d_func: zero-insertion → pad → rotated kernel → conv"""
    Win=x.shape[1]
    W_insert=Win+(Win-1)*(st[1]-1)
    y=np.zeros((Co,Wo),np.int64)
    for oc in range(Co):
        yc=np.zeros(Wo,np.int64)
        for ic in range(Ci):
            xc=x[ic]
            # Zero insertion
            xi=np.zeros(W_insert); xi[::st[1]]=xc
            # Padding: padarray(xi, [0,2], 0, 'both')
            xp=np.pad(xi,(2,2),'constant')
            # Kernel: weight(nIn,nOut,:,:) + rot90(.,2)
            wk=w[ic,oc] if w.ndim==4 else w[oc]
            wk_rot=np.rot90(wk.T, 2).ravel()
            for wi in range(Wo):
                xk=xp[wi:wi+ks[1]]
                yc[wi]+=np.sum(np.round(xk.astype(np.float64)*wk_rot*(2**Qr)).astype(np.int64))
        y[oc]=yc+b[oc]
    return y
def prelu(x,w):
    wf=w.ravel().astype(np.float64)/16384
    return np.where(x>=0, x, np.round(x.astype(np.float64)*wf)).astype(np.int64)

def bn_fixed(x,w,b,m,v,Qrv,Qrw):
    """BN: matching MATLAB bn_func(x,w,b,m,v, Qr1=var_shift, Qr2=weight_shift)"""
    y=np.zeros_like(x,np.int64)
    for c in range(x.shape[0]):
        wf=float(w[c]); bf=float(b[c]); mf=float(m[c]); vf=float(v[c])
        for i in range(x.shape[1]):
            xf=float(x[c,i])
            norm=int(round((xf-mf)*vf*(2**Qrv)))
            y[c,i]=int(round(norm*wf*(2**Qrw)))+int(bf)
    return y

def ln_fixed(x,w,b,Qr):
    """MATLAB ln_func: w,b are (F,C) per-frequency per-channel. Qr=-12 for RNN."""
    xf=x.ravel().astype(np.float64)/1048576.0
    mn=int(round(xf.mean()*1048576)); vr=int(round(1.0/np.sqrt(xf.var()+1e-8)*262144))
    y=np.zeros_like(x,np.int64)
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            nm=int(round((float(x[j,i])-mn)*vr/262144))
            y[j,i]=int(round(float(nm)*float(w[j,i])/(2**(-Qr))))+int(b[j,i])
    return y

def gru(x,H,hp,ih_w,ih_b,hh_w,hh_b):
    """GRU matching MATLAB GRU_module.m and BiGRU_module.m.
       x: (33,I) s32f20. hp: (33,H) for per-bin (Inter_RNN) or (H,) for single-state (BiGRU/Intra_RNN).
       Returns y:(33,H) s16f15, hp_out same shape as hp."""
    ih_w_q,_=FP(ih_w,'s16f12'); ih_b_q,_=FP(ih_b,'s16f10')
    hh_w_q,_=FP(hh_w,'s16f12'); hh_b_q,_=FP(hh_b,'s16f10')
    single_state = (hp.ndim == 1)  # BiGRU: h_prev = zeros(1,4)
    I=x.shape[1]; y=np.zeros((33,H),np.int64)
    hp_out = np.zeros_like(hp) if single_state else np.zeros_like(hp)
    for f in range(33):
        xt=x[f].astype(np.float64); hp_f=hp.astype(np.float64) if single_state else hp[f].astype(np.float64)
        rt=np.round(xt@ih_w_q[:H].T/4194304)+np.round(hp_f@hh_w_q[:H].T/131072)+ih_b_q[:H]+hh_b_q[:H]
        rs=1/(1+np.exp(-np.clip(rt.astype(np.float64)/1024,-88,88)))
        rq=np.clip(np.round(rs*32768),0,32767).astype(np.int64)
        zt=np.round(xt@ih_w_q[H:2*H].T/4194304)+np.round(hp_f@hh_w_q[H:2*H].T/131072)+ih_b_q[H:2*H]+hh_b_q[H:2*H]
        zs=1/(1+np.exp(-np.clip(zt.astype(np.float64)/1024,-88,88)))
        zq=np.clip(np.round(zs*32768),0,32767).astype(np.int64)
        ht=np.round(hp_f@hh_w_q[2*H:].T/131072)+hh_b_q[2*H:]
        nt=np.round(xt@ih_w_q[2*H:].T/4194304)+np.round(rq.astype(np.float64)*ht.astype(np.float64)/32768)+ih_b_q[2*H:]
        nth=np.tanh(np.clip(nt.astype(np.float64)/1024,-88,88))
        nq=np.clip(np.round(nth*32768),-32768,32767).astype(np.int64)
        hn=np.round((32768-zq).astype(np.float64)*nq.astype(np.float64)/32768)+np.round(zq.astype(np.float64)*hp_f/32768)
        if single_state: hp=hn.astype(np.int64); hp_out=hp
        else: hp[f]=hn.astype(np.int64); hp_out[f]=hn.astype(np.int64)
        y[f]=hn.astype(np.int64)
    return y,hp_out

# ═══ Pre/Post ═══
def mag_gen(r,i):
    mg=np.sqrt(r**2+i**2+1e-12); y=np.stack([mg,r,i]); y_q,_=FP(y,'s32f20'); return y_q

def BM_fixed(x,weight):
    wq,_=FP(weight,'u16f15'); y=np.zeros((3,129),np.int64)
    y[:,:65]=x[:,:65]
    y[:,65:]=np.round(x[:,65:].astype(np.float64)@wq.T.astype(np.float64)/32768).astype(np.int64)
    return y

def SFE(x):
    Ci,Wi=x.shape; xp=np.pad(x,((0,0),(1,1)))
    y=np.zeros((Ci*3,Wi),np.int64)
    for c in range(Ci):
        for k in range(3): y[c*3+k]=xp[c,k:k+Wi]
    return y

def MASK(m,r,i):
    """r,i: s32f20, m: s32f20 from BS → output s32f20. Divisor=2^20 matches s32f20*s32f20→s32f20"""
    yr=np.round(r.astype(np.float64)*m[0].astype(np.float64)/1048576)-np.round(i.astype(np.float64)*m[1].astype(np.float64)/1048576)
    yi=np.round(i.astype(np.float64)*m[0].astype(np.float64)/1048576)+np.round(r.astype(np.float64)*m[1].astype(np.float64)/1048576)
    return np.stack([yr,yi]).astype(np.int64)

def BS_fixed(x,weight):
    wq,_=FP(weight,'u16f15'); y=np.zeros((2,257),np.int64)
    y[:,:65]=x[:,:65]
    y[:,65:]=np.round(x[:,65:].astype(np.float64)@wq.T.astype(np.float64)/32768).astype(np.int64)
    return y

# ═══ TRA / DeTRA (Temporal Recurrent Attention) ═══
def TRA(x,hp,idx):
    """x: s32f20 (8,33) conv format → y: s32f20 (8,33)
       hp: s16f15 (16,) previous hidden state
       Aggregation: square + mean → GRU(16) → FC(16→8) + Sigmoid → attention
    """
    p=f'encoder_en_convs_{2+idx}'
    # Dequantize, square, average pool across frequency
    x_dq = x.astype(np.float64) / (2**20)
    x_sq = x_dq**2
    x_agg = np.mean(x_sq, axis=1)  # (8,) mean across 33 freq bins
    x_agg_q, _ = FP(x_agg, 'u32f20')  # (8,) u32f20

    # GRU with nHidden=16
    g_ih=W(f'{p}_tra_att_gru_weight_ih_l0').T; g_ih_b=W(f'{p}_tra_att_gru_bias_ih_l0').ravel()
    g_hh=W(f'{p}_tra_att_gru_weight_hh_l0').T; g_hh_b=W(f'{p}_tra_att_gru_bias_hh_l0').ravel()

    ih_w_q,_=FP(g_ih,'s16f13'); ih_b_q,_=FP(g_ih_b,'s32f20')
    hh_w_q,_=FP(g_hh,'s16f13'); hh_b_q,_=FP(g_hh_b,'s32f20')

    # Single-timestep GRU with H=16, I=8
    xt=x_agg_q.astype(np.float64)  # (8,)
    hp_f=hp.astype(np.float64)     # (16,)

    # R gate
    rt=np.round(xt@ih_w_q[:,:16]/(2**13))+np.round(hp_f@hh_w_q[:,:16]/(2**8))+ih_b_q[:16]+hh_b_q[:16]
    rs=1/(1+np.exp(-np.clip(rt/(2**20),-88,88)))
    rq=np.clip(np.round(rs*32768),0,32767).astype(np.int64)
    # Z gate
    zt=np.round(xt@ih_w_q[:,16:32]/(2**13))+np.round(hp_f@hh_w_q[:,16:32]/(2**8))+ih_b_q[16:32]+hh_b_q[16:32]
    zs=1/(1+np.exp(-np.clip(zt/(2**20),-88,88)))
    zq=np.clip(np.round(zs*32768),0,32767).astype(np.int64)
    # N gate
    ht=np.round(hp_f@hh_w_q[:,32:48]/(2**8))+hh_b_q[32:48]
    nt=np.round(xt@ih_w_q[:,32:48]/(2**13))+np.round(rq.astype(np.float64)*ht/(2**15))+ih_b_q[32:48]
    nth=np.tanh(np.clip(nt/(2**20),-88,88))
    nq=np.clip(np.round(nth*32768),-32768,32767).astype(np.int64)
    # Hidden update
    hp_new=np.round((32768-zq).astype(np.float64)*nq.astype(np.float64)/32768)+np.round(zq.astype(np.float64)*hp_f/32768)
    hp_new=hp_new.astype(np.int64)  # s16f15 (16,)

    # FC: (16,) → (8,) + Sigmoid
    fc_w=W(f'{p}_tra_att_fc_weight').T; fc_b=W(f'{p}_tra_att_fc_bias').ravel()
    fc_w_q,_=FP(fc_w,'s16f13'); fc_b_q,_=FP(fc_b,'s32f20')
    x_fc=np.round(hp_new.astype(np.float64)@fc_w_q/(2**8))+fc_b_q  # s32f20 (8,)
    x_fc_dq=x_fc/(2**20)
    x_act=1/(1+np.exp(-np.clip(x_fc_dq,-88,88)))
    x_act_q,_=FP(x_act,'u16f15')  # (8,) u16f15

    # Apply attention: y = x * act (broadcast across frequency)
    y=np.round(x.astype(np.float64)*x_act_q.astype(np.float64).reshape(-1,1)/(2**15)).astype(np.int64)
    return y, hp_new

def DeTRA(x,hp,idx):
    """Same as TRA but loads from decoder_de_convs_{idx}"""
    p=f'decoder_de_convs_{idx}'
    x_dq = x.astype(np.float64) / (2**20)
    x_sq = x_dq**2
    x_agg = np.mean(x_sq, axis=1)
    x_agg_q, _ = FP(x_agg, 'u32f20')

    g_ih=W(f'{p}_tra_att_gru_weight_ih_l0').T; g_ih_b=W(f'{p}_tra_att_gru_bias_ih_l0').ravel()
    g_hh=W(f'{p}_tra_att_gru_weight_hh_l0').T; g_hh_b=W(f'{p}_tra_att_gru_bias_hh_l0').ravel()
    ih_w_q,_=FP(g_ih,'s16f13'); ih_b_q,_=FP(g_ih_b,'s32f20')
    hh_w_q,_=FP(g_hh,'s16f13'); hh_b_q,_=FP(g_hh_b,'s32f20')

    xt=x_agg_q.astype(np.float64); hp_f=hp.astype(np.float64)
    rt=np.round(xt@ih_w_q[:,:16]/(2**13))+np.round(hp_f@hh_w_q[:,:16]/(2**8))+ih_b_q[:16]+hh_b_q[:16]
    rs=1/(1+np.exp(-np.clip(rt/(2**20),-88,88)))
    rq=np.clip(np.round(rs*32768),0,32767).astype(np.int64)
    zt=np.round(xt@ih_w_q[:,16:32]/(2**13))+np.round(hp_f@hh_w_q[:,16:32]/(2**8))+ih_b_q[16:32]+hh_b_q[16:32]
    zs=1/(1+np.exp(-np.clip(zt/(2**20),-88,88)))
    zq=np.clip(np.round(zs*32768),0,32767).astype(np.int64)
    ht=np.round(hp_f@hh_w_q[:,32:48]/(2**8))+hh_b_q[32:48]
    nt=np.round(xt@ih_w_q[:,32:48]/(2**13))+np.round(rq.astype(np.float64)*ht/(2**15))+ih_b_q[32:48]
    nth=np.tanh(np.clip(nt/(2**20),-88,88))
    nq=np.clip(np.round(nth*32768),-32768,32767).astype(np.int64)
    hp_new=np.round((32768-zq).astype(np.float64)*nq.astype(np.float64)/32768)+np.round(zq.astype(np.float64)*hp_f/32768)
    hp_new=hp_new.astype(np.int64)  # s16f15 (16,)

    fc_w=W(f'{p}_tra_att_fc_weight').T; fc_b=W(f'{p}_tra_att_fc_bias').ravel()
    fc_w_q,_=FP(fc_w,'s16f13'); fc_b_q,_=FP(fc_b,'s32f20')
    x_fc=np.round(hp_new.astype(np.float64)@fc_w_q/(2**8))+fc_b_q
    x_fc_dq=x_fc/(2**20)
    x_act=1/(1+np.exp(-np.clip(x_fc_dq,-88,88)))
    x_act_q,_=FP(x_act,'u16f15')

    y=np.round(x.astype(np.float64)*x_act_q.astype(np.float64).reshape(-1,1)/(2**15)).astype(np.int64)
    return y, hp_new

# ═══ Conv Block ═══
def Conv_block(x,idx):
    p=f'encoder_en_convs_{idx}'
    conv_w=W(f'{p}_conv_weight'); conv_b=W(f'{p}_conv_bias').ravel()
    bn_w=W(f'{p}_bn_weight').ravel(); bn_b=W(f'{p}_bn_bias').ravel()
    bn_m=W(f'{p}_bn_running_mean').ravel(); bn_v=1/np.sqrt(W(f'{p}_bn_running_var').ravel()+1e-5)
    prelu_w=W(f'{p}_act_weight').ravel()
    if idx==0:
        conv_w_q,_=FP(conv_w,'s32f18'); conv_b_q,_=FP(conv_b,'s32f20')
        bn_w_q,_=FP(bn_w,'u16f14'); bn_b_q,_=FP(bn_b,'s32f20')
        bn_m_q,_=FP(bn_m,'s32f20'); bn_v_q,_=FP(bn_v,'u16f14'); prelu_w_q,_=FP(prelu_w,'s16f14')
        Cin,Cout,Qr,Qr_bn_w,Qr_bn_v=9,16,-18,-14,-14
        Win=129; Wout=65
        y_conv=conv2d(x,conv_w_q,conv_b_q,Cin,Cout,1,Wout,[1,5],[1,2],[0,2],Qr)
    else:  # Conv1: groups=2, different Q formats
        conv_w_q,_=FP(conv_w,'s16f13'); conv_b_q,_=FP(conv_b,'s32f20')
        bn_w_q,_=FP(bn_w,'u16f14'); bn_b_q,_=FP(bn_b,'s32f20')
        bn_m_q,_=FP(bn_m,'s32f20'); bn_v_q,_=FP(bn_v,'u16f10'); prelu_w_q,_=FP(prelu_w,'s16f14')
        Cin,Cout,Qr,Qr_bn_w,Qr_bn_v=8,8,-13,-10,-14
        Win=65; Wout=33
        # Group 1
        y1=conv2d(x[:8],conv_w_q[:8],conv_b_q[:8],Cin,Cout,1,Wout,[1,5],[1,2],[0,2],Qr)
        # Group 2
        y2=conv2d(x[8:16],conv_w_q[8:16],conv_b_q[8:16],Cin,Cout,1,Wout,[1,5],[1,2],[0,2],Qr)
        y_conv=np.concatenate([y1,y2])
    y_bn=bn_fixed(y_conv,bn_w_q,bn_b_q,bn_m_q,bn_v_q,Qr_bn_v,Qr_bn_w)
    return prelu(y_bn,prelu_w_q)

# ═══ GT-Conv ═══
def GT_Conv(x,cp,hp,dil,idx):
    x1=x[:8]; x2=x[8:16]; x1_sfe=SFE(x1)
    p=f'encoder_en_convs_{2+idx}'
    # P-Conv-0: s16f13, Qr=-13, BN(-13,-14), PReLU
    pc0_w=W(f'{p}_point_conv1_weight'); pc0_b=W(f'{p}_point_conv1_bias').ravel()
    bn1_w=W(f'{p}_point_bn1_weight').ravel(); bn1_b=W(f'{p}_point_bn1_bias').ravel()
    bn1_m=W(f'{p}_point_bn1_running_mean').ravel(); bn1_v=1/np.sqrt(W(f'{p}_point_bn1_running_var').ravel()+1e-5)
    act1_w=W(f'{p}_point_act_weight').ravel()
    pc0_w_q,_=FP(pc0_w,'s16f13'); pc0_b_q,_=FP(pc0_b,'s32f20')
    bn1_w_q,_=FP(bn1_w,'u16f14'); bn1_b_q,_=FP(bn1_b,'s32f20')
    bn1_m_q,_=FP(bn1_m,'s32f20'); bn1_v_q,_=FP(bn1_v,'u16f13'); act1_w_q,_=FP(act1_w,'s16f14')
    y_pc0=conv2d(x1_sfe,pc0_w_q,pc0_b_q,24,16,1,33,[1,1],[1,1],[0,0],-13)
    y_bn1=bn_fixed(y_pc0,bn1_w_q,bn1_b_q,bn1_m_q,bn1_v_q,-13,-14)
    y_act1=prelu(y_bn1,act1_w_q)
    # DD-Conv: s16f13, Qr=-13, 2D time buffer
    dc_w=W(f'{p}_depth_conv_weight'); dc_b=W(f'{p}_depth_conv_bias').ravel()
    dbn_w=W(f'{p}_depth_bn_weight').ravel(); dbn_b=W(f'{p}_depth_bn_bias').ravel()
    dbn_m=W(f'{p}_depth_bn_running_mean').ravel(); dbn_v=1/np.sqrt(W(f'{p}_depth_bn_running_var').ravel()+1e-5)
    dact_w=W(f'{p}_depth_act_weight').ravel()
    dc_w_q,_=FP(dc_w,'s16f13'); dc_b_q,_=FP(dc_b,'s32f20')
    dbn_w_q,_=FP(dbn_w,'u16f14'); dbn_b_q,_=FP(dbn_b,'s32f20')
    dbn_m_q,_=FP(dbn_m,'s32f20'); dbn_v_q,_=FP(dbn_v,'u16f10'); dact_w_q,_=FP(dact_w,'s16f14')
    x_inp=np.concatenate([cp,y_act1.reshape(16,1,33)],axis=1)
    y_dc=ddconv2d(x_inp,dc_w_q,dc_b_q,dil,-13)
    y_dbn=bn_fixed(y_dc,dbn_w_q,dbn_b_q,dbn_m_q,dbn_v_q,-10,-14)
    y_dact=prelu(y_dbn,dact_w_q)
    cp_out=x_inp[:,1:,:]  # remove oldest frame, keep buffer size constant
    # P-Conv-1: s16f13, Qr=-13, BN(-14,-14), no PReLU
    pc1_w=W(f'{p}_point_conv2_weight'); pc1_b=W(f'{p}_point_conv2_bias').ravel()
    bn2_w=W(f'{p}_point_bn2_weight').ravel(); bn2_b=W(f'{p}_point_bn2_bias').ravel()
    bn2_m=W(f'{p}_point_bn2_running_mean').ravel(); bn2_v=1/np.sqrt(W(f'{p}_point_bn2_running_var').ravel()+1e-5)
    pc1_w_q,_=FP(pc1_w,'s16f13'); pc1_b_q,_=FP(pc1_b,'s32f20')
    bn2_w_q,_=FP(bn2_w,'u16f14'); bn2_b_q,_=FP(bn2_b,'s32f20')
    bn2_m_q,_=FP(bn2_m,'s32f20'); bn2_v_q,_=FP(bn2_v,'u16f14')
    y_pc1=conv2d(y_dact,pc1_w_q,pc1_b_q,16,8,1,33,[1,1],[1,1],[0,0],-13)
    y_bn2=bn_fixed(y_pc1,bn2_w_q,bn2_b_q,bn2_m_q,bn2_v_q,-14,-14)
    y_tra,hp_new=TRA(y_bn2,hp,idx)
    y=np.zeros((16,33),np.int64); y[0::2]=y_tra; y[1::2]=x2
    return y,cp_out,hp_new

# ═══ Encoder ═══
def Encoder(x,cp,hp):
    """cp: list of 3 independent buffers [cp0(16,2,33), cp1(16,4,33), cp2(16,10,33)]"""
    y0=Conv_block(x,0); y1=Conv_block(y0,1)
    y2,cp[0],hp[0]=GT_Conv(y1,cp[0],hp[0],1,0)
    y3,cp[1],hp[1]=GT_Conv(y2,cp[1],hp[1],2,1)
    y4,cp[2],hp[2]=GT_Conv(y3,cp[2],hp[2],5,2)
    return y0,y1,y2,y3,y4,cp,hp

# ═══ Intra/Inter RNN ═══
def Intra_RNN(x,idx):
    p=f'dpgrnn{idx}'; nH=4
    r1_ih=W(f'{p}_intra_rnn_rnn1_weight_ih_l0'); r1_ih_b=W(f'{p}_intra_rnn_rnn1_bias_ih_l0').ravel()
    r1_hh=W(f'{p}_intra_rnn_rnn1_weight_hh_l0'); r1_hh_b=W(f'{p}_intra_rnn_rnn1_bias_hh_l0').ravel()
    r1_rih=W(f'{p}_intra_rnn_rnn1_weight_ih_l0_reverse'); r1_rih_b=W(f'{p}_intra_rnn_rnn1_bias_ih_l0_reverse').ravel()
    r1_rhh=W(f'{p}_intra_rnn_rnn1_weight_hh_l0_reverse'); r1_rhh_b=W(f'{p}_intra_rnn_rnn1_bias_hh_l0_reverse').ravel()
    r2_ih=W(f'{p}_intra_rnn_rnn2_weight_ih_l0'); r2_ih_b=W(f'{p}_intra_rnn_rnn2_bias_ih_l0').ravel()
    r2_hh=W(f'{p}_intra_rnn_rnn2_weight_hh_l0'); r2_hh_b=W(f'{p}_intra_rnn_rnn2_bias_hh_l0').ravel()
    r2_rih=W(f'{p}_intra_rnn_rnn2_weight_ih_l0_reverse'); r2_rih_b=W(f'{p}_intra_rnn_rnn2_bias_ih_l0_reverse').ravel()
    r2_rhh=W(f'{p}_intra_rnn_rnn2_weight_hh_l0_reverse'); r2_rhh_b=W(f'{p}_intra_rnn_rnn2_bias_hh_l0_reverse').ravel()
    fc_w=W(f'{p}_intra_fc_weight').T; fc_b=W(f'{p}_intra_fc_bias').ravel()
    ln_w=W(f'{p}_intra_ln_weight'); ln_b=W(f'{p}_intra_ln_bias')
    xT=x.T; x1=xT[:,:8]; x2=xT[:,8:16]; h0=np.zeros(nH,np.int64)
    g1_fwd,_=gru(x1,nH,h0.copy(),r1_ih,r1_ih_b,r1_hh,r1_hh_b)
    x1_rev=x1[::-1]; g1_rev,_=gru(x1_rev,nH,h0.copy(),r1_rih,r1_rih_b,r1_rhh,r1_rhh_b)
    x1_gru=np.concatenate([g1_fwd,g1_rev[::-1]],1)
    g2_fwd,_=gru(x2,nH,h0.copy(),r2_ih,r2_ih_b,r2_hh,r2_hh_b)
    x2_rev=x2[::-1]; g2_rev,_=gru(x2_rev,nH,h0.copy(),r2_rih,r2_rih_b,r2_rhh,r2_rhh_b)
    x2_gru=np.concatenate([g2_fwd,g2_rev[::-1]],1)
    x_gru=np.concatenate([x1_gru,x2_gru],1)
    fc_w_q,_=FP(fc_w,'s16f13'); fc_b_q,_=FP(fc_b,'s32f20')
    x_fc=np.round(x_gru.astype(np.float64)@fc_w_q.astype(np.float64)/256)+fc_b_q
    ln_w_q,_=FP(ln_w,'s16f12'); ln_b_q,_=FP(ln_b,'s32f20')
    y_out=ln_fixed(x_fc,ln_w_q,ln_b_q,-12); return (xT+y_out).T

def Inter_RNN(x,ip,idx):
    p=f'dpgrnn{idx}'; nH=8
    r1_ih=W(f'{p}_inter_rnn_rnn1_weight_ih_l0'); r1_ih_b=W(f'{p}_inter_rnn_rnn1_bias_ih_l0').ravel()
    r1_hh=W(f'{p}_inter_rnn_rnn1_weight_hh_l0'); r1_hh_b=W(f'{p}_inter_rnn_rnn1_bias_hh_l0').ravel()
    r2_ih=W(f'{p}_inter_rnn_rnn2_weight_ih_l0'); r2_ih_b=W(f'{p}_inter_rnn_rnn2_bias_ih_l0').ravel()
    r2_hh=W(f'{p}_inter_rnn_rnn2_weight_hh_l0'); r2_hh_b=W(f'{p}_inter_rnn_rnn2_bias_hh_l0').ravel()
    fc_w=W(f'{p}_inter_fc_weight').T; fc_b=W(f'{p}_inter_fc_bias').ravel()
    ln_w=W(f'{p}_inter_ln_weight'); ln_b=W(f'{p}_inter_ln_bias')
    xT=x.T; x1=xT[:,:8]; x2=xT[:,8:16]; h0=ip.copy()
    g1_fwd,hp1=gru(x1,nH,h0[:,:8],r1_ih,r1_ih_b,r1_hh,r1_hh_b)
    g2_fwd,hp2=gru(x2,nH,h0[:,8:16],r2_ih,r2_ih_b,r2_hh,r2_hh_b)
    x_gru=np.concatenate([g1_fwd,g2_fwd],1); ip_new=np.concatenate([hp1,hp2],1)
    fc_w_q,_=FP(fc_w,'s16f13'); fc_b_q,_=FP(fc_b,'s32f20')
    x_fc=np.round(x_gru.astype(np.float64)@fc_w_q.astype(np.float64)/256)+fc_b_q
    ln_w_q,_=FP(ln_w,'s16f12'); ln_b_q,_=FP(ln_b,'s32f20')
    y_out=ln_fixed(x_fc,ln_w_q,ln_b_q,-12); return (xT+y_out).T,ip_new

# ═══ ddconv2d (2D dilated depthwise conv — encoder, NO rot90) ═══
def ddconv2d(x_inp,weight_q,bias_q,dil,Qr):
    """MATLAB ddconv2d_func: 2D time×freq dilated conv, NO kernel rotation.
    x_inp: (C, Hin, Win) time buffer. weight_q: (C,1,3,3). bias_q: (C,).
    dil: time dilation. Qr: requant shift.
    """
    C,Hin,Win=x_inp.shape; Wout=Win; y=np.zeros((C,Wout),np.int64)
    for c in range(C):
        xc=x_inp[c]; k=weight_q[c,0]
        # Dilated kernel (no rot90 for forward conv)
        kd=np.zeros((2*dil+1,3),np.float64)
        kd[0,:]=k[0,:]; kd[dil,:]=k[1,:]; kd[2*dil,:]=k[2,:]
        # Pad input: padarray(xc, [0,1], 0, 'both')
        x_pad=np.pad(xc,((0,0),(1,1)),'constant')
        for w in range(Wout):
            s=bias_q[c]
            for h in range(2*dil+1):
                if h>=Hin: continue
                for kw in range(3):
                    iw=w+kw
                    if 0<=iw<x_pad.shape[1]:
                        s+=int(round(float(x_pad[h,iw])*kd[h,kw]*(2**Qr)))
            y[c,w]=s
    return y

# ═══ ddtconv2d (2D dilated transposed depthwise conv — decoder, WITH rot90) ═══
def ddtconv2d(x_inp,weight_q,bias_q,dil,Qr):
    """MATLAB ddtconv2d_func: 2D time×freq dilated conv with rot90 kernel.
    x_inp: (C, Hin, Win) time buffer. weight_q: (C,1,3,3). bias_q: (C,).
    dil: time dilation. Qr: requant shift.
    """
    C,Hin,Win=x_inp.shape; Wout=Win; y=np.zeros((C,Wout),np.int64)
    for c in range(C):
        xc=x_inp[c]; k=weight_q[c,0]
        kd=np.zeros((2*dil+1,3),np.float64)
        kd[0,:]=k[0,:]; kd[dil,:]=k[1,:]; kd[2*dil,:]=k[2,:]
        k_rot=np.rot90(kd,2)
        x_pad=np.pad(xc,((0,0),(1,1)),'constant')
        for w in range(Wout):
            s=bias_q[c]
            for h in range(2*dil+1):
                if h>=Hin: continue
                for kw in range(3):
                    iw=w+kw
                    if 0<=iw<x_pad.shape[1]:
                        s+=int(round(float(x_pad[h,iw])*k_rot[h,kw]*(2**Qr)))
            y[c,w]=s
    return y

# ═══ GT-DeConv ═══
def GT_DeConv(x,sk,cp,hp,dil,idx):
    x=x+sk; x1=x[:8]; x2=x[8:16]  # skip connection at module input
    p=f'decoder_de_convs_{idx}'
    # P-DeConv-0: s16f13, Qr=-13, BN(-14,-14), PReLU
    pc0_w=W(f'{p}_point_conv1_weight'); pc0_b=W(f'{p}_point_conv1_bias').ravel()
    bn1_w=W(f'{p}_point_bn1_weight').ravel(); bn1_b=W(f'{p}_point_bn1_bias').ravel()
    bn1_m=W(f'{p}_point_bn1_running_mean').ravel(); bn1_v=1/np.sqrt(W(f'{p}_point_bn1_running_var').ravel()+1e-5)
    act1_w=W(f'{p}_point_act_weight').ravel()
    pc0_w_q,_=FP(pc0_w,'s16f13'); pc0_b_q,_=FP(pc0_b,'s32f20')
    bn1_w_q,_=FP(bn1_w,'u16f14'); bn1_b_q,_=FP(bn1_b,'s32f20')
    bn1_m_q,_=FP(bn1_m,'s32f20'); bn1_v_q,_=FP(bn1_v,'u16f14'); act1_w_q,_=FP(act1_w,'s16f14')
    x1_sfe=SFE(x1)
    y_pc0=ptconv2d(x1_sfe,pc0_w_q,pc0_b_q,24,16,1,33,-13)
    y_bn1=bn_fixed(y_pc0,bn1_w_q,bn1_b_q,bn1_m_q,bn1_v_q,-14,-14)
    y_act1=prelu(y_bn1,act1_w_q)
    # DD-DeConv: s16f12, Qr=-12, 2D time buffer, export-script algorithm
    dc_w=W(f'{p}_depth_conv_weight'); dc_b=W(f'{p}_depth_conv_bias').ravel()
    dbn_w=W(f'{p}_depth_bn_weight').ravel(); dbn_b=W(f'{p}_depth_bn_bias').ravel()
    dbn_m=W(f'{p}_depth_bn_running_mean').ravel(); dbn_v=1/np.sqrt(W(f'{p}_depth_bn_running_var').ravel()+1e-5)
    dact_w=W(f'{p}_depth_act_weight').ravel()
    dc_w_q,_=FP(dc_w,'s16f12'); dc_b_q,_=FP(dc_b,'s32f20')
    dbn_w_q,_=FP(dbn_w,'u16f14'); dbn_b_q,_=FP(dbn_b,'s32f20')
    dbn_m_q,_=FP(dbn_m,'s32f20'); dbn_v_q,_=FP(dbn_v,'u16f12'); dact_w_q,_=FP(dact_w,'s16f14')
    # Build time buffer + 2D dilated transposed conv (matching ddtconv2d_func)
    x_inp=np.concatenate([cp,y_act1.reshape(16,1,33)],axis=1)
    y_dc=ddtconv2d(x_inp,dc_w_q,dc_b_q,dil,-12)
    y_dbn=bn_fixed(y_dc,dbn_w_q,dbn_b_q,dbn_m_q,dbn_v_q,-12,-14)
    y_dact=prelu(y_dbn,dact_w_q)+sk
    cp_out=x_inp[:,1:,:]  # shift out oldest frame
    # P-DeConv-1: s16f13, Qr=-13, BN(-14,-14), no PReLU
    pc1_w=W(f'{p}_point_conv2_weight'); pc1_b=W(f'{p}_point_conv2_bias').ravel()
    bn2_w=W(f'{p}_point_bn2_weight').ravel(); bn2_b=W(f'{p}_point_bn2_bias').ravel()
    bn2_m=W(f'{p}_point_bn2_running_mean').ravel(); bn2_v=1/np.sqrt(W(f'{p}_point_bn2_running_var').ravel()+1e-5)
    pc1_w_q,_=FP(pc1_w,'s16f13'); pc1_b_q,_=FP(pc1_b,'s32f20')
    bn2_w_q,_=FP(bn2_w,'u16f14'); bn2_b_q,_=FP(bn2_b,'s32f20')
    bn2_m_q,_=FP(bn2_m,'s32f20'); bn2_v_q,_=FP(bn2_v,'u16f14')
    y_pc1=ptconv2d(y_dact,pc1_w_q,pc1_b_q,16,8,1,33,-13)
    y_bn2=bn_fixed(y_pc1,bn2_w_q,bn2_b_q,bn2_m_q,bn2_v_q,-14,-14)
    y_tra,hp_new=DeTRA(y_bn2,hp,idx)
    y=np.zeros((16,33),np.int64); y[0::2]=y_tra; y[1::2]=x2
    return y,cp_out,hp_new

# ═══ DeConv Block ═══
def DeConv_block(x,sk,idx):
    """idx=1: DeConv1(groups=2,s16f13,Qr=-13,PReLU). idx=0: DeConv0(no groups,s32f18,Qr=-18,Tanh→s16f15)"""
    p=f'decoder_de_convs_{4-idx}'
    conv_w=W(f'{p}_conv_weight'); conv_b=W(f'{p}_conv_bias').ravel()
    bn_w=W(f'{p}_bn_weight').ravel(); bn_b=W(f'{p}_bn_bias').ravel()
    bn_m=W(f'{p}_bn_running_mean').ravel(); bn_v=1/np.sqrt(W(f'{p}_bn_running_var').ravel()+1e-5)
    try: act_w=W(f'{p}_act_weight').ravel(); has_act=True
    except: act_w=None; has_act=False

    if sk is not None: x_sk=x+sk
    else: x_sk=x

    if idx==1:  # DeConv1: groups=2, s16f13, Qr=-13, PReLU
        conv_w_q,_=FP(conv_w,'s16f13'); conv_b_q,_=FP(conv_b,'s32f20')
        bn_w_q,_=FP(bn_w,'u16f14'); bn_b_q,_=FP(bn_b,'s32f20')
        bn_m_q,_=FP(bn_m,'s32f20'); bn_v_q,_=FP(bn_v,'u16f14')
        act_w_q,_=(FP(act_w,'s16f14') if has_act else (None,None))
        Wout_t=(x_sk.shape[1]-1)*2+5-4
        y1=tconv2d(x_sk[:8],conv_w_q[:8],conv_b_q[:8],8,8,1,Wout_t,[1,5],[1,2],[0,2],-13)
        y2=tconv2d(x_sk[8:16],conv_w_q[8:16],conv_b_q[8:16],8,8,1,Wout_t,[1,5],[1,2],[0,2],-13)
        y_conv=np.concatenate([y1,y2])
        y_bn=bn_fixed(y_conv,bn_w_q,bn_b_q,bn_m_q,bn_v_q,-14,-14)
        y=prelu(y_bn,act_w_q)
    else:  # DeConv0: no groups, s32f18, Qr=-18, Tanh→s16f15
        conv_w_q,_=FP(conv_w,'s32f18'); conv_b_q,_=FP(conv_b,'s32f20')
        bn_w_q,_=FP(bn_w,'u16f14'); bn_b_q,_=FP(bn_b,'s32f20')
        bn_m_q,_=FP(bn_m,'s32f20'); bn_v_q,_=FP(bn_v,'u16f14')
        Wout_t=(x_sk.shape[1]-1)*2+5-4
        Cin=conv_w_q.shape[0]; Cout=conv_b_q.size
        y_conv=tconv2d(x_sk,conv_w_q,conv_b_q,Cin,Cout,1,Wout_t,[1,5],[1,2],[0,2],-18)
        y_bn=bn_fixed(y_conv,bn_w_q,bn_b_q,bn_m_q,bn_v_q,-14,-14)
        # Tanh + quantize to s16f15, then scale to s32f20 for downstream compatibility
        y_bn_f=y_bn.astype(np.float64)/(2**20)
        y_tanh=np.tanh(y_bn_f)
        y_q,_=FP(y_tanh,'s16f15')
        y=(y_q.astype(np.int64)<<5)  # s16f15 → s32f20: multiply by 2^(20-15)=32
    return y

# ═══ Decoder ═══
def Decoder(x,y0,y1,y2,y3,y4,cp,hp):
    """cp: list of 3 independent buffers [cp0(16,10,33), cp1(16,4,33), cp2(16,2,33)] matching MATLAB"""
    y_d0,cp[0],hp[0]=GT_DeConv(x,y4,cp[0],hp[0],5,0)
    y_d1,cp[1],hp[1]=GT_DeConv(y_d0,y3,cp[1],hp[1],2,1)
    y_d2,cp[2],hp[2]=GT_DeConv(y_d1,y2,cp[2],hp[2],1,2)
    y_d3=DeConv_block(y_d2,y1,1)
    y=DeConv_block(y_d3,y0,0)
    return y,cp,hp

# ═══ Main inference ═══
def infer_frame(spec, state):
    """Process one STFT frame, return enhanced CRM (2,257) float32"""
    cp_enc, hp_enc, inter1, inter2, cp_dec, hp_dec, erbfc_w, ierbfc_w = state

    x = mag_gen(spec.real.astype(np.float64), spec.imag.astype(np.float64))
    x_bm = BM_fixed(x, erbfc_w)
    x_sfe = SFE(x_bm)

    y0,y1,y2,y3,y4,cp_enc,hp_enc = Encoder(x_sfe, cp_enc, hp_enc)
    y_rnn1=Intra_RNN(y4, 1)
    y_rnn2,inter2=Inter_RNN(y_rnn1, inter1, 1)
    y_rnn3=Intra_RNN(y_rnn2, 2)
    y_rnn4,inter2=Inter_RNN(y_rnn3, inter2, 2)
    y_dec,cp_dec,hp_dec = Decoder(y_rnn4, y0,y1,y2,y3,y4, cp_dec, hp_dec)

    y_bs = BS_fixed(y_dec, ierbfc_w)
    y_mask = MASK(y_bs, x[1], x[2])

    new_state = (cp_enc, hp_enc, inter1, inter2, cp_dec, hp_dec, erbfc_w, ierbfc_w)
    return y_mask.astype(np.float32) / (2**20), new_state

def init_state():
    erbfc_w = W('erb_erb_fc_weight')
    ierbfc_w = W('erb_ierb_fc_weight')
    # MATLAB: 3 independent per-GT buffers matching max_hist_len = [2, 4, 10]
    cp_enc = [np.zeros((16,2,33),np.int64), np.zeros((16,4,33),np.int64), np.zeros((16,10,33),np.int64)]
    hp_enc = np.zeros((3,16),np.int64)
    cp_dec = [np.zeros((16,10,33),np.int64), np.zeros((16,4,33),np.int64), np.zeros((16,2,33),np.int64)]
    hp_dec = np.zeros((3,16),np.int64)
    return (cp_enc, hp_enc,
            np.zeros((33,16),np.int64), np.zeros((33,16),np.int64),
            cp_dec, hp_dec,
            erbfc_w, ierbfc_w)

print("gtcrn_full.py loaded successfully.")
