from .models import *

class Lit4dVarNet_CROSCIM_Supervised(Lit4dVarNet_CROSCIM):
    """
    Extension of Lit4dVarNet_CROSCIM with support for numerical model inputs.
    Adds models_vars alongside satellite observations and covariates.
    """
    
    def __init__(
        self,
        models_vars=None,
        norm_stats_models=None,
        **kwargs
    ):
        """
        Args:
            models_vars: list of model variable names, e.g., ["SIC", "SIT"]
            norm_stats_models: dict of normalization stats for model variables
            **kwargs: passed to parent Lit4dVarNet_CROSCIM
        """
        # ✅ Store models-specific config BEFORE calling super().__init__
        self.models_vars = models_vars or []
        self.norm_stats_models = norm_stats_models or {}
        
        # ✅ Call parent init - this will set self.satellite_vars, self.covariates, etc.
        super().__init__(**kwargs)
        
        # ✅ NOW we can safely modify active_sources (after parent has created it)
        if not hasattr(self, 'active_sources'):
            # Parent didn't create it, so create it ourselves
            self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        # Add models to active sources if configured
        if self.models_vars and 'models' not in self.active_sources:
            self.active_sources.append('models')
        
        # ✅ Build model input variable names
        self.input_vars_models = [f"models_{var}" for var in self.models_vars]
        
        # ✅ Update input_vars_satellite if not already set by parent
        if not hasattr(self, 'input_vars_satellite'):
            self.input_vars_satellite = [
                f"{source}_{var}" 
                for source, vars in self.satellite_vars.items() 
                for var in vars
            ]
        
        # ✅ Update total input vars list (satellite + models + covariates)
        self.input_vars_all = self.input_vars_satellite + self.input_vars_models + self.covariates
        
        print(f"\n{'='*60}")
        print(f"Lit4dVarNet_CROSCIM_Supervised initialized:")
        print(f"{'='*60}")
        print(f"  Models vars: {self.models_vars}")
        print(f"  Input vars models: {self.input_vars_models}")
        print(f"  Active sources: {self.active_sources}")
        print(f"  Total input vars: {len(self.input_vars_all)}")
        print(f"    - Satellite: {len(self.input_vars_satellite)}")
        print(f"    - Models: {len(self.input_vars_models)}")
        print(f"    - Covariates: {len(self.covariates)}")
        print(f"{'='*60}\n")
    
    def normalize_data(self, batch_dict):
        """
        Extends parent normalize_data to also normalize model variables.
        """
        # ✅ First normalize satellite vars and covariates (parent behavior)
        batch_dict = super().normalize_data(batch_dict)
        
        # ✅ Then normalize model variables
        for key, batch in batch_dict.items():
            for var in self.models_vars:
                var_name = f"models_{var}"
                if hasattr(batch, var_name):
                    stats = self.norm_stats_models.get(var, {})
                    tensor = getattr(batch, var_name)
                    
                    if stats.get('type') == 'minmax':
                        normalized = (tensor - stats['min']) / (stats['max'] - stats['min'])
                    elif stats.get('type') == 'zscore':
                        normalized = (tensor - stats['mean']) / stats['std']
                    else:
                        normalized = tensor  # No normalization
                    
                    setattr(batch, var_name, normalized)
        
        return batch_dict
    
    def prepare_input_tensor(self, batch):
        """
        Extends parent prepare_input_tensor to include model variables.
        Order: satellite vars → model vars → covariates
        """
        tensors = []
        
        # Add satellite variables
        for var_name in self.input_vars_satellite:
            if hasattr(batch, var_name):
                tensors.append(getattr(batch, var_name))
            else:
                print(f"Warning: {var_name} not found in batch")
        
        # Add model variables
        for var_name in self.input_vars_models:
            if hasattr(batch, var_name):
                tensors.append(getattr(batch, var_name))
            else:
                print(f"Warning: {var_name} not found in batch")
        
        # Add covariates
        for cov in self.covariates:
            if hasattr(batch, cov):
                tensors.append(getattr(batch, cov))
            else:
                print(f"Warning: {cov} not found in batch")
        
        # Concatenate along channel dimension
        if tensors:
            input_tensor = torch.cat(tensors, dim=1)  # (B, C, T, Y, X)
            return input_tensor
        else:
            raise ValueError("No input tensors found in batch")