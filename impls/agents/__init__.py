from .bc import BCAgent
from .crl import CRLAgent
from .gcbc import GCBCAgent
from .gcivl import GCIVLAgent
from .gciql import GCIQLAgent
from .hiql import HIQLAgent
#from impls.agents.ppo import PPOAgent
from .qrl import QRLAgent
from .sac import SACAgent
from .cmd import CMDAgent
from .tmd import TMDAgent
from .tmd_dc import TMDDCAgent
from .tmd_dqc import TMDDQCAgent
from .tmd_qc import TMDQCAgent
#from impls.agents.tra import TRAAgent

# Registry keyed by module basename (e.g. server --agent tmd_dqc -> agents.tmd_dqc).
agents = dict(
    bc=BCAgent,
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gcivl=GCIVLAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    #ppo=PPOAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cmd=CMDAgent,
    tmd=TMDAgent,
    tmd_dc=TMDDCAgent,
    tmd_dqc=TMDDQCAgent,
    tmd_qc=TMDQCAgent,
    #tra=TRAAgent,
)
