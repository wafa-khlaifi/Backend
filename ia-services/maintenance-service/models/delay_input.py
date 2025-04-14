from pydantic import BaseModel

class DelayInput(BaseModel):
    assetnum: str
    location: str
    worktype: str
    calcpriority: float
    schedstart: str  # format ISO (ex: "2025-04-01T08:30:00")
