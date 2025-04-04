#!/bin/bash

# Create a backup of the current agents/__init__.py
cp agents/__init__.py agents/__init__.py.bak

# Copy all agent implementation files from impls/agents to agents
cp impls/agents/*.py agents/

# Update the imports in the agents/__init__.py file
cat > agents/__init__.py << 'EOF'
from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.cmd import CMDAgent
from agents.gcivl import GCIVLAgent

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
EOF

# Update imports in all agent files
for file in agents/*.py; do
  if [ "$file" != "agents/__init__.py" ]; then
    sed -i 's/from impls\.utils/from impls.utils/g' "$file"
  fi
done

echo "Agent files moved and imports updated successfully!" 