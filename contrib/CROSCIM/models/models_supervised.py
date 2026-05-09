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
        #  Store models-specific config BEFORE calling super().__init__
        self.models_vars = models_vars or []
        self.norm_stats_models = norm_stats_models or {}
        
        #  Call parent init - this will set self.satellite_vars, self.covariates, etc.
        super().__init__(**kwargs)
        
        #  NOW we can safely modify active_sources (after parent has created it)
        if not hasattr(self, 'active_sources'):
            # Parent didn't create it, so create it ourselves
            self.active_sources = [src for src, vars in self.satellite_vars.items() if vars]
        
        # Add models to active sources if configured
        if self.models_vars and 'models' not in self.active_sources:
            self.active_sources.append('models')
        
        #  Build model input variable names
        self.input_vars_models = [f"models_{var}" for var in self.models_vars]
        
        #  Update input_vars_satellite if not already set by parent
        if not hasattr(self, 'input_vars_satellite'):
            self.input_vars_satellite = [
                f"{source}_{var}" 
                for source, vars in self.satellite_vars.items() 
                for var in vars
            ]
        
        #  Update total input vars list (satellite + models + covariates)
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
        #  First normalize satellite vars and covariates (parent behavior)
        batch_dict = super().normalize_data(batch_dict)
        
        #  Then normalize model variables
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

    def apply_models_mask_to_batch(self, batch):
        """
        Si des variables cibles sont de type `models_XXX` (ex: models_SIC, models_SIT),
        elles contiennent des NaN là où le modèle numérique ne fournit pas de données
        (typiquement le masque terre/côte).

        Ce masque est extrait (union des NaN sur tous les pas de temps) puis appliqué
        à toutes les variables satellites du batch : les pixels invalides dans le modèle
        numérique deviennent NaN dans les observations satellites aussi.

        Cela garantit que le solveur ne voit jamais de pixels pour lesquels la cible
        est indisponible, évitant des artefacts aux bordures du domaine modèle.

        Returns
        -------
        batch : namedtuple du même type, avec les variables satellites masquées.
        models_mask : BoolTensor (B, T, H, W), True là où les modèles sont valides.
        """
        batch_dict = batch._asdict()

        # ── 1. Construire le masque union de tous les models_XXX ──────────
        # On cherche les champs du batch dont le nom commence par "models_"
        models_mask = None  # True = pixel valide dans le modèle numérique
        for var_name, tensor in batch_dict.items():
            if not var_name.startswith("models_"):
                continue
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            # valid_here : True là où AU MOINS un pas de temps est valide
            # (on prend l'union temporelle pour être conservateur)
            valid_here = tensor.isfinite().any(dim=1, keepdim=True)  # (B, 1, H, W)
            valid_here = valid_here.expand_as(tensor)                  # (B, T, H, W)
            models_mask = valid_here if models_mask is None else (models_mask & valid_here)

        if models_mask is None:
            # Pas de variable models_ → rien à faire
            return batch, None

        # ── 2. Appliquer le masque à toutes les variables satellites ──────
        new_dict = dict(batch_dict)
        satellite_var_names = [
            f"{src}_{var}"
            for src, vars in self.satellite_vars.items()
            for var in vars
        ]
        for var_name in satellite_var_names:
            if var_name not in new_dict:
                continue
            t = new_dict[var_name]
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            # Mettre à NaN les pixels hors du domaine modèle
            new_dict[var_name] = t.where(models_mask, torch.tensor(float('nan'), device=t.device, dtype=t.dtype))

        return type(batch)(**new_dict), models_mask

    def format_batch_for_solver(self, batch, include_masks=False, res=None):
        """
        Étend le parent en appliquant d'abord le masque des variables models_XXX
        à toutes les variables satellites, puis délègue au parent.
        """
        batch, _ = self.apply_models_mask_to_batch(batch)
        return super().format_batch_for_solver(batch, include_masks=include_masks, res=res)

