from abc import ABC, abstractmethod
from typing import Tuple

import IRModule


class IGetTvmModel(ABC):
    @abstractmethod
    def get(self) -> Tuple[IRModule, dict]:
        pass
