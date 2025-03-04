from haystack import component, default_from_dict, default_to_dict
from typing import Dict, Any

@component
class ExampleComponent:
    def __init__(self, param1):
        self.param1 = param1
    
    def _do_something(self, input:str):
        return f"{input} with param {self.param1}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        Check haytack.utils for more details on handling serialization for other types 

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            self.param1
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExampleComponent":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        return default_from_dict(cls, data)

    
    @component.output_types(response=str)
    def run(self, input: str):
        return {"response": self._do_something(input)}