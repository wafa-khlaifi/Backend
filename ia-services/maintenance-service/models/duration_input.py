from pydantic import BaseModel, Field
from typing import Optional

class DurationInput(BaseModel):
    delay_days: float = Field(..., ge=0)
    assetnum: str
    location: str
    worktype: str
    frequency: Optional[float] = None
    regularhrs: Optional[float] = None
    linecost: Optional[float] = None
