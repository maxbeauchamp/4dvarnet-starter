"""Custom OmegaConf resolvers for the project."""
from omegaconf import OmegaConf


def register_custom_resolvers():
    """Register all custom OmegaConf resolvers."""
    
    def python_eval_resolver(expression):
        """
        Safely evaluate Python expressions in OmegaConf.
        Converts lists/dicts to OmegaConf objects.
        """
        # Safe builtins for evaluation
        safe_globals = {
            "__builtins__": {},
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "range": range,
            "list": list,
            "dict": dict,
            "int": int,
            "float": float,
            "str": str,
        }
        
        try:
            result = eval(expression, safe_globals, {})
            
            # Convert Python types to OmegaConf types
            if isinstance(result, list):
                return OmegaConf.create(result)  # ✅ Convert list to ListConfig
            elif isinstance(result, dict):
                return OmegaConf.create(result)  # ✅ Convert dict to DictConfig
            elif isinstance(result, (int, float, str, bool)) or result is None:
                return result  # Primitives are fine
            else:
                raise ValueError(f"Unsupported type: {type(result)}")
                
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {e}")
    
    # Register the resolver
    OmegaConf.register_new_resolver(
        "python_eval", 
        python_eval_resolver,
        replace=True
    )
    
    print("✅ Custom OmegaConf resolver 'python_eval' registered")


# Auto-register when module is imported
register_custom_resolvers()
