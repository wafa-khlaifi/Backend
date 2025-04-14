from pydantic import BaseModel

class PriorityInput(BaseModel):
    estlabcost: float
    estmatcost: float
    worktype: str
    status: str
    estdur: float