TOOL_REGISTRY = {}

def register_tool(name):
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator

def get_tool(name):
    return TOOL_REGISTRY.get(name)

def list_tools():
    return list(TOOL_REGISTRY.keys())
