from datetime import datetime
from pydantic import BaseModel

class AgentStep(BaseModel):
    agent_name: str
    action: str
    result: str
    timestamp: datetime
