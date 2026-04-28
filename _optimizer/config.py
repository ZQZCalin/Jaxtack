# Optimizer config classes.
from pydantic import BaseModel

class BaseOptimizerConfig(BaseModel):
    name: str
