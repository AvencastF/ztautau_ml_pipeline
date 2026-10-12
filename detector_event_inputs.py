"""Saved reconstructed globals and explicitly versioned angular supervision.

Truth is used only in angular_targets, never reconstructed_conditions.
"""
import json,math
import numpy as np
from detector_inputs import scalar

GLOBAL_VALUES=('charged_thrust_axis_x','charged_thrust_axis_y','charged_thrust_axis_z','charged_isolation_degrees')
GLOBAL_NAMES=(*GLOBAL_VALUES,*(n+'__available' for n in GLOBAL_VALUES))
TARGET_VERSION='saved_resolved_hpc_v1_dr03_truth_tau_minus_visible_angles_v1'

def decode(v):return json.loads(v) if isinstance(v,str) else v

def reconstructed_conditions(rows):
    values=np.zeros((len(rows),4),np.float64);valid=np.zeros_like(values,dtype=bool)
    for i,r in enumerate(rows):
        a=decode(r.get('axis'))
        if isinstance(a,list) and len(a)==3 and all(isinstance(x,(int,float)) and not isinstance(x,bool) for x in a) and np.isfinite(a).all() and np.isclose(np.linalg.norm(a),1.,rtol=1e-6,atol=1e-8):
            values[i,:3]=a;valid[i,:3]=True
        x=r.get('value.isolation')
        if r.get('valid.isolation') is True and isinstance(x,(int,float)) and math.isfinite(x):values[i,3]=x;valid[i,3]=True
    return values,valid

def fit_conditions(rows):
    v,m=reconstructed_conditions(rows);count=m.sum(0);mean=np.divide((v*m).sum(0),count,out=np.zeros(4),where=count>0)
    variance=np.divide(((v-mean)**2*m).sum(0),count,out=np.zeros(4),where=count>0);std=np.sqrt(variance);std[std==0]=1
    return {'names':list(GLOBAL_VALUES),'mean':mean.tolist(),'std':std.tolist(),'count':count.tolist()}

def conditions(rows,normal):
    v,m=reconstructed_conditions(rows);supported=np.asarray(normal['count'])>0;usable=m&supported
    x=np.where(usable,(v-np.asarray(normal['mean']))/np.asarray(normal['std']),0.)
    return np.concatenate((x,usable.astype(float)),1).astype(np.float32),usable.any(1,keepdims=True)

def angles(p):
    p=np.asarray(p,float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('undefined direction')
    pt=np.hypot(p[0],p[1]);norm=np.linalg.norm(p)
    if norm<=0 or pt<=0:raise ValueError('undefined direction')
    return np.array([math.atan2(pt,p[2]),math.atan2(p[1],p[0])])

def angular_targets(rows,labels,classes,config):
    out=np.zeros((len(rows),2,2),np.float32);mask=np.zeros((len(rows),2),bool);reasons=[]
    supported=config['NeutrinoPrediction']['Ztautau']
    for j,r in enumerate(rows):
        reason='unsupported_class'
        if labels[j]<0 or classes[labels[j]] not in supported:reasons.append(reason);continue
        try:
            payload=decode(r['particle_detector']);particles=payload['particles'];leads=decode(r['leading_indices'])
            by_index={p['main_index']:p for p in particles}
            if set(by_index)!=set(range(len(particles))) or len(by_index)!=len(particles):raise ValueError('MAIN alignment malformed')
            if len(leads)!=2 or len(set(leads))!=2 or any(type(i)!=int or not 0<=i<len(particles) for i in leads):raise ValueError('invalid leading associations')
            fields=[by_index[i]['main']['fields'] for i in leads]
            charges=[scalar(f.get('charge',{})) for f in fields]
            if any(not c[1] or c[0] not in (-1,1) or (c[2] and not c[3]) for c in charges) or charges[0][0]==charges[1][0]:raise ValueError('charge ordering unavailable')
            order=sorted(range(2),key=lambda k:-charges[k][0])
            momenta=[]
            for f in fields:
                xyz=[scalar(f.get('momentum.'+c,{})) for c in 'xyz']
                if any(not x[1] or (x[2] and not x[3]) for x in xyz):raise ValueError('leading momentum unavailable')
                momenta.append(np.array([x[0] for x in xyz]))
            photons=decode(r['photon_associations'])
            if photons.get('working_point')!='existing_input_resolved_hpc_v1' or photons.get('valid',{}).get('selected_photon_count') is not True:raise ValueError('photon membership unavailable')
            selected=[p for p in photons['records'] if p.get('resolved_hpc_pass') is True]
            if len({p.get('main_index') for p in selected})!=len(selected) or len(selected)!=photons['values']['selected_photon_count']:raise ValueError('photon membership malformed')
            visible=[p.copy() for p in momenta]
            for photon in selected:
                p=np.asarray(photon['momentum_gev'],float);a=angles(p);eta=-math.log(math.tan(a[0]/2))
                for k,lead in enumerate(momenta):
                    b=angles(lead);dphi=(a[1]-b[1]+math.pi)%(2*math.pi)-math.pi
                    if math.hypot(eta+math.log(math.tan(b[0]/2)),dphi)<.3:visible[k]+=p
            truth=decode(r['truth_decay']);gen=decode(r['generator_particles']);prefix='sDST_LUJ_GenParticles/sDST_LUJ_GenParticles.'
            for side,k in enumerate(order):
                name=('plus','minus')[side];idx=truth[name]['tau_index']
                if type(idx)!=int or idx<0:raise ValueError('truth tau index invalid')
                if gen[prefix+'PDG'][idx]!=(-15 if side==0 else 15):raise ValueError('truth tau charge mismatch')
                tau=[gen[prefix+'momentum.'+c][idx] for c in 'xyz'];delta=angles(tau)-angles(visible[k]);delta[1]=(delta[1]+math.pi)%(2*math.pi)-math.pi
                out[j,side]=delta;mask[j,side]=True
            reason='available'
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as e:
            out[j]=0;mask[j]=False;reason=str(e)
        reasons.append(reason)
    return out,mask,reasons
