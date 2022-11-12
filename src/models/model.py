import inspect

class Model():
    def __init__(self, **kwargs):
        self._name = None
        self._description = None
        self._kwargs = kwargs

    def _run(self, **kwargs):
        raise NotImplementedError

    def run(self, data):
        filtered_mydict = {
            k: v for k, v in data.items() if k in [p.name for p in inspect.signature(self._run).parameters.values()]
        }

        return self._run(**filtered_mydict)

    def __str__(self):
        return self._name

    def __description__(self):
        return self._description