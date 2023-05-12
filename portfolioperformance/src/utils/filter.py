import inspect

# This is the function that is used to filter the parameters passed to a function
def filterParams(params, obj, func):
    f = getattr(obj, func)

    filteredDict = {
        k: v for k, v in params.items() if k in [p.name for p in inspect.signature(f).parameters.values()]
    }

    return filteredDict
