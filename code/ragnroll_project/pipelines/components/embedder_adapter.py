from haystack import component

@component
class EmbedderAdapter:
    @component.output_types(text=str)
    def run(self, query: str) -> str:
        return {"text": query}
