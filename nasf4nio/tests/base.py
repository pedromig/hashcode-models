from typing import Callable, Optional, Any, TypeVar

Problem = TypeVar("Problem")

# Injector
def property_test(test: Callable[[Problem, Optional[int]], None]) -> Problem:
    def decorator(cls):
        setattr(cls, test.__name__, test)
        return cls
    return decorator