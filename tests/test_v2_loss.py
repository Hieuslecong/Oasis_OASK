import pytest
torch=pytest.importorskip("torch")
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import oasis_rc_student_loss_v2

def test_v2_corrupted_ranking_has_student_gradient_only():
 c=OASISRCv2Critic(width=4)
 for p in c.parameters():p.requires_grad_(False)
 x=torch.rand(2,3,32,32);y=torch.zeros(2,1,32,32);y[:,:,8:24,15:17]=1;l=torch.zeros(2,1,32,32,requires_grad=True);p=l.sigmoid();w=y.flip(-1)
 with torch.no_grad():gt=c(x,y);co=c(x,w)
 loss,t=oasis_rc_student_loss_v2(c(x,p),gt,co,p,y);loss.backward();assert torch.isfinite(loss) and torch.isfinite(l.grad).all() and l.grad.abs().sum()>0;assert all(z.grad is None for z in c.parameters());assert {"rank_gt","rank_corrupted","fp","e_pred","e_gt","e_corrupted"}.issubset(t)
def test_v2_critic_output_contract():
 o=OASISRCv2Critic(width=4)(torch.rand(1,3,32,32),torch.rand(1,1,32,32));assert o["semantic"].shape==(1,3,32,32);assert o["mismatch"].shape==(1,1,32,32);assert o["pair"].shape==(1,1)
