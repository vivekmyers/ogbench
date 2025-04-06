from impls.agents.crl import CRLAgent
from impls.agents.gcbc import GCBCAgent
from impls.agents.gciql import GCIQLAgent
from impls.agents.hiql import HIQLAgent
#from impls.agents.ppo import PPOAgent
from impls.agents.qrl import QRLAgent
from impls.agents.sac import SACAgent
from impls.agents.cmd import CMDAgent
#from impls.agents.tra import TRAAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    #ppo=PPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cmd=CMDAgent,
    #tra=TRAAgent,
)
