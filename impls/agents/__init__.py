from .bc import BCAgent
from .crl import CRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .hiql import HIQLAgent
#from impls.agents.ppo import PPOAgent
from .qrl import QRLAgent
from .sac import SACAgent
from .cmd import CMDAgent
#from impls.agents.tra import TRAAgent

agents = dict(
    bc=BCAgent,
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
