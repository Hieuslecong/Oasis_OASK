import copy,hashlib
from types import SimpleNamespace
import pytest,torch
from oasis_cycle_aosk.aosk import oriented_consistency_loss
from oasis_cycle_aosk.models import MultiScaleLightweightSegmenter
from oasis_cycle_aosk.train_oasis_rc_v2 import augment,build_targets,load_student_init,make_corrupted_mask,make_generator,sha256_file,validate_loaded_critic
from oasis_rc_v2.checkpoint import CHECKPOINT_SCHEMA,EXPERIMENT_ID,METHOD_VERSION
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import segmentation_loss,oasis_rc_student_loss_v2

def flat(loss,params,retain=False):
 g=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True);return torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1) for p,x in zip(params,g)])
def fixture():
 torch.manual_seed(123);s=MultiScaleLightweightSegmenter(width=4);x=torch.randn(2,3,32,32);y=torch.zeros(2,1,32,32);y[:,:,14:18,4:28]=1;return s,x,y
def test_s0_equals_s2_when_lambda_aosk_zero():
 s,x,y=fixture();p=[z for z in s.parameters() if z.requires_grad];l=s(x);seg=segmentation_loss(l,y);a=oriented_consistency_loss(l,x,y);assert torch.equal(flat(seg,p,True),flat(seg+0*a,p))
def test_s1_equals_s3_when_lambda_aosk_zero():
 s,x,y=fixture();p=[z for z in s.parameters() if z.requires_grad];c=OASISRCv2Critic(width=4).eval()
 for z in c.parameters():z.requires_grad_(False)
 l=s(x);seg=segmentation_loss(l,y);pred=l.sigmoid();wrong,_=make_corrupted_mask(y,generator=make_generator(torch.device("cpu"),17))
 with torch.no_grad():gt=c(x,y);co=c(x,wrong)
 rc,e=oasis_rc_student_loss_v2(c(x,pred),gt,co,pred,y);a=oriented_consistency_loss(l,x,y);assert torch.equal(flat(seg+.001*rc,p,True),flat(seg+.001*rc+0*a,p));assert all(k in e for k in ("e_pred","e_gt","e_corrupted","delta_pred_gt","delta_pred_corrupted"))
def test_rc_corruption_rng_does_not_change_augmentation_sequence():
 x=torch.linspace(-1,1,2*3*16*16).reshape(2,3,16,16);y=torch.zeros(2,1,16,16);y[:,:,6:10,3:13]=1;c=make_generator(torch.device("cpu"),777);r=make_generator(torch.device("cpu"),777);cg=make_generator(torch.device("cpu"),999);c1=augment(x.clone(),y.clone(),c);c2=augment(x.clone(),y.clone(),c);r1=augment(x.clone(),y.clone(),r);make_corrupted_mask(y,generator=cg);r2=augment(x.clone(),y.clone(),r);assert torch.equal(c1[0],r1[0]) and torch.equal(c2[0],r2[0])
def test_noop_target_is_pair_valid():
 m=torch.zeros(2,1,8,8);_,mm,pv=build_targets(m,torch.zeros_like(m));assert float(mm.sum())==0 and torch.equal(pv,torch.ones_like(pv))
def test_two_step_zero_rc_equivalence():
 base,x,y=fixture();a=copy.deepcopy(base);b=copy.deepcopy(base);c=OASISRCv2Critic(width=4).eval()
 for p in c.parameters():p.requires_grad_(False)
 oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4);ga=make_generator(torch.device("cpu"),101);gb=make_generator(torch.device("cpu"),101);gc=make_generator(torch.device("cpu"),202)
 for _ in range(2):
  xa,ya=augment(x.clone(),y.clone(),ga);xb,yb=augment(x.clone(),y.clone(),gb);la=segmentation_loss(a(xa),ya);oa.zero_grad();la.backward();oa.step();log=b(xb);seg=segmentation_loss(log,yb);pred=log.sigmoid();wrong,_=make_corrupted_mask(yb,generator=gc)
  with torch.no_grad():gt=c(xb,yb);co=c(xb,wrong)
  rc,_=oasis_rc_student_loss_v2(c(xb,pred),gt,co,pred,yb);lb=seg+0*rc;ob.zero_grad();lb.backward();ob.step()
 for (n1,t1),(n2,t2) in zip(a.state_dict().items(),b.state_dict().items()):assert n1==n2 and torch.equal(t1,t2),n1
def test_student_init_seed_mismatch_is_rejected(tmp_path):
 s=MultiScaleLightweightSegmenter(width=4);setattr(s,"_oasis_width",4);p=tmp_path/"i.pt";torch.save({"student":s.state_dict(),"student_kind":"multiscale","student_width":4,"seed":1337},p)
 with pytest.raises(ValueError,match="seed mismatch"):load_student_init(s,p,2027)
def test_critic_provenance_and_schema_fail_closed(tmp_path):
 m=tmp_path/"m";m.write_text("x");args=SimpleNamespace(manifest=str(m),normal_fraction=.25,normal_critic_weight=1.0);cfg={"seed":1337,"image_size":256};saved={"checkpoint_schema":CHECKPOINT_SCHEMA,"experiment_id":EXPERIMENT_ID,"method_version":METHOD_VERSION,"critic":{},"width":8,"config":cfg,"manifest_file_sha256":sha256_file(m),"normal_fraction":.25,"normal_critic_weight":1.0};validate_loaded_critic(saved,args,cfg);bad=dict(saved);bad.pop("checkpoint_schema")
 with pytest.raises(ValueError,match="legacy checkpoint rejected"):validate_loaded_critic(bad,args,cfg)
def test_sha256_exact(tmp_path):
 p=tmp_path/"x";p.write_bytes(b"a");assert sha256_file(p)==hashlib.sha256(b"a").hexdigest()
