from .crl import CRLAgent
from .gcbc import GCBCAgent
from .gciql import GCIQLAgent
from .hiql import HIQLAgent
from .qrl import QRLAgent
from .sac import SACAgent
from .cmd import CMDAgent
from .gcivl import GCIVLAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    cmd=CMDAgent,
    gcivl=GCIVLAgent,
) 