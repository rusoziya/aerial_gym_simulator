from __future__ import annotations

"""
Static Camera Influence Tracker for RL Training
Activation-based monitoring to verify if neural network uses static camera data.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CompleteObservationInfluenceTracker:
    """
    Enhanced tracker that analyzes the influence of ALL observation components on the neural network's encoded features.
    
    Extends the previous camera-only analysis to cover the complete 150D observation space:
    - Drone position (3D)
    - Static camera pose (6D) 
    - Drone orientation (3D)
    - Kinematic state (6D)
    - Action history (4D)
    - Drone camera VAE (64D)
    - Static camera VAE (64D)
    
    Provides correlation, magnitude, and variance analysis for each component.
    """
    
    def __init__(self, model, config: dict[str, object]):
        self.model = model
        self.config = config
        self.enabled = True
        self.step_count = 0
        self.forward_pass_count = 0  # Track forward passes separately
        self.hook_handles = []
        
        # Complete observation component mapping (150D total)
        self.obs_components = {
            'drone_position': (0, 3, "🎯", "Drone absolute position in world coordinates"),
            'static_camera_pos': (3, 6, "📍", "Static camera position relative to drone"),
            'static_camera_orient': (6, 9, "🧭", "Static camera orientation relative to drone"),
            'drone_orientation': (9, 12, "✈️", "Drone full orientation (roll, pitch, yaw)"),
            'drone_linear_vel': (12, 15, "⚡", "Drone linear velocity in body frame"),
            'drone_angular_vel': (15, 18, "🌀", "Drone angular velocity in body frame"),
            'drone_actions': (18, 22, "🎮", "Previous drone actions (action history)"),
            'drone_camera_vae': (22, 86, "📷", "Drone camera VAE encoded features"),
            'static_camera_vae': (86, 150, "📹", "Static camera VAE encoded features"),
        }
        
        # Storage for analysis data
        self.activation_history = {
            component: {
                'correlations': [],
                'magnitudes': [],
                'variances': []
            } for component in self.obs_components.keys()
        }
        
        # Try to attach hooks in order of preference
        self._attach_hooks()
        
    def _attach_hooks(self) -> None:
        """Attach forward hooks to capture activations at the encoder output."""
        hook_targets = [
            ('encoder.encoders.obs', 'Obs Encoder (non-compiled)'),
            ('obs_normalizer', 'Observation Normalizer'),
            ('encoder', 'Multi Input Encoder'),
            ('core', 'RNN Core'),
        ]
        
        for target_path, target_name in hook_targets:
            try:
                target_module = self.model
                for attr in target_path.split('.'):
                    target_module = getattr(target_module, attr)
                
                # Check if this is a ScriptModule (which doesn't support hooks)
                if hasattr(target_module, '_c') or 'ScriptModule' in str(type(target_module)):
                    logger.warning(f"⚠️ Skipping {target_name} - ScriptModule doesn't support hooks")
                    continue
                
                # Try to attach hook
                handle = target_module.register_forward_hook(self._activation_hook)
                self.hook_handles.append(handle)
                logger.warning(f"✅ Complete observation tracker attached to {target_name}")
                return  # Success - only need one hook
                
            except AttributeError as e:
                logger.warning(f"⚠️ Could not access {target_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Failed to attach to {target_name}: {e}")
                continue
                
        # If all targets failed, try to find any non-ScriptModule components
        logger.warning("🔍 Searching for non-ScriptModule components...")
        for name, module in self.model.named_modules():
            if (hasattr(module, '_c') or 'ScriptModule' in str(type(module)) or
                'RecursiveScriptModule' in str(type(module))):
                continue
                
            # Skip modules that are too basic
            if any(skip_type in str(type(module)) for skip_type in ['Linear', 'ELU', 'Identity', 'GRU']):
                continue
                
            try:
                handle = module.register_forward_hook(self._activation_hook)
                self.hook_handles.append(handle)
                logger.warning(f"✅ Complete observation tracker attached to non-ScriptModule: {name} ({type(module)})")
                return  # Success
            except Exception as e:
                continue
                
        logger.warning("❌ Failed to attach complete observation tracker - no suitable non-ScriptModule hook points found")
        logger.warning("💡 ScriptModules don't support hooks - this is a PyTorch limitation")
        self.enabled = False
        
    def _activation_hook(self, module, input, output) -> None:
        """Hook that captures activations and analyzes complete observation influence.
        Robust to modules that accept/return dicts or tuples (SF encoders, custom fusion encoders).
        """
        if not self.enabled:
            return

        def _extract_obs_tensor(obj) -> None:
            # Try common wrapper types: tuple(list(dict(...)))
            try:
                import torch as _torch
                # Direct tensor
                if isinstance(obj, _torch.Tensor):
                    return obj
                # Tuple/list: search first tensor or dict field
                if isinstance(obj, (tuple, list)) and len(obj) > 0:
                    return _extract_obs_tensor(obj[0])
                # Dict: typical Sample Factory keys
                if isinstance(obj, dict):
                    if 'obs' in obj and isinstance(obj['obs'], _torch.Tensor):
                        return obj['obs']
                    if 'observations' in obj and isinstance(obj['observations'], _torch.Tensor):
                        return obj['observations']
                    # Fallback: first tensor value
                    for v in obj.values():
                        if isinstance(v, _torch.Tensor):
                            return v
                return None
            except Exception:
                return None

        def _extract_feat_tensor(obj) -> None:
            try:
                import torch as _torch
                if isinstance(obj, _torch.Tensor):
                    return obj
                if isinstance(obj, (tuple, list)) and len(obj) > 0:
                    return _extract_feat_tensor(obj[0])
                if isinstance(obj, dict):
                    for key in ('encoding', 'x', 'h', 'features', 'out'):
                        if key in obj and isinstance(obj[key], _torch.Tensor):
                            return obj[key]
                    for v in obj.values():
                        if isinstance(v, _torch.Tensor):
                            return v
                return None
            except Exception:
                return None

        try:
            observations = _extract_obs_tensor(input)
            encoded_features = _extract_feat_tensor(output)

            # Validate
            if observations is None or not hasattr(observations, 'dim'):
                return
            if encoded_features is None or not hasattr(encoded_features, 'dim'):
                return
            if observations.dim() != 2 or observations.shape[1] < 81:
                # Expect at least the standard 81D or the 150D gate observation
                return

            # Analyze influence for all observation components
            self._analyze_complete_observation_influence(observations, encoded_features)

        except Exception as e:
            if self.step_count < 5:  # Only log errors for first few steps
                logger.warning(f"🔧 Hook analysis error: {e}")
                
    def _analyze_complete_observation_influence(self, observations: torch.Tensor, encoded_features: torch.Tensor) -> None:
        """Analyze the influence of all observation components on encoded features."""
        batch_size = observations.shape[0]
        
        if batch_size < 2:  # Need at least 2 samples for correlation
            return
            
        # Increment forward pass counter
        self.forward_pass_count += 1
        
        step_data = {}
        
        # Analyze each observation component
        for component_name, (start, end, emoji, description) in self.obs_components.items():
            component_data = observations[:, start:end]
            
            # Calculate metrics for this component
            magnitude = torch.norm(component_data, dim=1).mean().item()
            variance = torch.var(component_data).item()
            
            # Calculate correlation with encoded features
            correlation = self._calculate_component_correlation(component_data, encoded_features)
            
            # Store metrics
            step_data[component_name] = {
                'correlation': correlation,
                'magnitude': magnitude,
                'variance': variance,
                'emoji': emoji,
                'description': description
            }
            
            # Add to history
            self.activation_history[component_name]['correlations'].append(correlation)
            self.activation_history[component_name]['magnitudes'].append(magnitude)
            self.activation_history[component_name]['variances'].append(variance)
            
        # Debug output (reduced frequency) - only every 50 forward passes
        if self.forward_pass_count <= 3 or self.forward_pass_count % 50 == 0:
            self._log_step_debug(step_data)
            
    def _calculate_component_correlation(self, component_data: torch.Tensor, encoded_features: torch.Tensor) -> float:
        """Calculate correlation between component data and encoded features."""
        if encoded_features.shape[0] < 2:
            return 0.0
            
        correlations = []
        
        # Average component features across dimensions for correlation calculation
        if component_data.shape[1] > 1:
            component_avg = component_data.mean(dim=1)
        else:
            component_avg = component_data.squeeze(1)
            
        # Sample subset of encoded features for efficiency
        max_features_to_check = min(encoded_features.shape[1], 32)
        
        for i in range(max_features_to_check):
            encoded_feature_i = encoded_features[:, i]
            
            # Check for sufficient variance
            if torch.var(component_avg) > 1e-6 and torch.var(encoded_feature_i) > 1e-6:
                try:
                    corr_matrix = torch.corrcoef(torch.stack([component_avg, encoded_feature_i]))
                    corr_value = corr_matrix[0, 1].item()
                    
                    if not torch.isnan(torch.tensor(corr_value)):
                        correlations.append(abs(corr_value))
                        
                except Exception:
                    continue
                    
        return np.mean(correlations) if correlations else 0.0
        
    def _log_step_debug(self, step_data: Dict) -> None:
        """Log debug information for current step."""
        
        # Determine training phase
        if self.step_count == 0:
            phase = "📊 DATA COLLECTION PHASE"
        else:
            phase = f"🎯 TRAINING PHASE (SF Step {self.step_count})"
            
        # logger.warning(f"🔧 COMPLETE OBSERVATION ANALYSIS - {phase}")  # Suppressed verbose periodic debug output
        # logger.warning(f"   Sample Factory Training Step: {self.step_count} | Forward Pass: {self.forward_pass_count}")  # Suppressed verbose periodic debug output
        
        # Sort components by correlation strength
        sorted_components = sorted(step_data.items(), key=lambda x: x[1]['correlation'], reverse=True)
        
        for component_name, data in sorted_components:
            emoji = data['emoji']
            corr = data['correlation']
            mag = data['magnitude']
            var = data['variance']
            # logger.warning(f"   {emoji} {component_name:20s}: Corr={corr:.3f}, Mag={mag:.1f}, Var={var:.3f}")  # Suppressed verbose periodic debug output
        
        # logger.warning("")  # Add spacing (suppressed)
        
    def step(self) -> None:
        """Update step counter - now synced externally with Sample Factory."""
        # Note: step_count is now set directly by the training script
        # to match Sample Factory's actual training step count
        if self.step_count % 10 == 0:  # Log every 10 training steps
            # logger.warning(f"📈 Training milestone: Step {self.step_count} reached (Total forward passes: {self.forward_pass_count})")  # Suppressed verbose periodic debug output
            pass

    def should_log(self) -> bool:
        """Determine if metrics should be logged this step."""
        return self.step_count > 0 and self.step_count % self.config.get('log_interval', 100) == 0
        
    def get_logging_metrics(self) -> Dict[str, float]:
        """Get metrics for logging to wandb/tensorboard."""
        if not self.activation_history:
            return {}
            
        metrics = {}
        
        for component_name in self.obs_components.keys():
            history = self.activation_history[component_name]
            
            if history['correlations']:
                # Recent average (last 10 samples)
                recent_corr = np.mean(history['correlations'][-10:])
                recent_mag = np.mean(history['magnitudes'][-10:])
                recent_var = np.mean(history['variances'][-10:])
                
                metrics[f'obs_influence/{component_name}_correlation'] = recent_corr
                metrics[f'obs_influence/{component_name}_magnitude'] = recent_mag
                metrics[f'obs_influence/{component_name}_variance'] = recent_var
                
        return metrics
        
    def print_analysis_summary(self) -> None:
        """Print comprehensive analysis summary."""
        if not any(self.activation_history[comp]['correlations'] for comp in self.obs_components.keys()):
            logger.warning("📊 No analysis data available yet")
            return
            
        logger.warning("================================================================================")
        logger.warning("📊 COMPLETE OBSERVATION INFLUENCE ANALYSIS")
        logger.warning("================================================================================")
        logger.warning(f"📊 Analysis based on {self.step_count} Sample Factory training steps and {self.forward_pass_count} forward passes")
        logger.warning("    (Note: Sample Factory processes multiple rollout steps per training step)")
        logger.warning("")
        
        # Calculate averages for all components
        component_stats = {}
        for component_name, (start, end, emoji, description) in self.obs_components.items():
            history = self.activation_history[component_name]
            
            if history['correlations']:
                avg_correlation = np.mean(history['correlations'])
                avg_magnitude = np.mean(history['magnitudes'])
                avg_variance = np.mean(history['variances'])
                sample_count = len(history['correlations'])
                
                component_stats[component_name] = {
                    'correlation': avg_correlation,
                    'magnitude': avg_magnitude,
                    'variance': avg_variance,
                    'samples': sample_count,
                    'emoji': emoji,
                    'description': description,
                    'dimensions': end - start
                }
                
        # Sort by correlation strength
        sorted_stats = sorted(component_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)
        
        logger.warning("📊 INDIVIDUAL OBSERVATION COMPONENT INFLUENCE:")
        logger.warning("")
        
        # Display each component individually with clear formatting
        total_correlation = sum(stats['correlation'] for _, stats in component_stats.items())
        
        for rank, (component_name, stats) in enumerate(sorted_stats, 1):
            corr = stats['correlation']
            mag = stats['magnitude']
            var = stats['variance']
            samples = stats['samples']
            emoji = stats['emoji']
            desc = stats['description']
            dims = stats['dimensions']
            
            # Calculate percentage of total influence
            influence_pct = (corr / total_correlation * 100) if total_correlation > 0 else 0
            
            # Determine importance level
            if corr > 0.2:
                importance = "🔥 CRITICAL"
                bar = "████████████"
            elif corr > 0.1:
                importance = "📊 SIGNIFICANT"
                bar = "████████░░░░"
            elif corr > 0.05:
                importance = "⚡ MODERATE"
                bar = "████░░░░░░░░"
            else:
                importance = "❌ MINIMAL"
                bar = "██░░░░░░░░░░"
                
            logger.warning(f"{rank:2d}. {emoji} {component_name.upper():<20s} | Correlation: {corr:.4f} ({influence_pct:5.1f}%)")
            logger.warning(f"     {importance} | {bar} | Dimensions: {dims}D | Samples: {samples}")
            logger.warning(f"     Magnitude: {mag:8.1f} | Variance: {var:8.3f}")
            logger.warning(f"     Description: {desc}")
            logger.warning("")
            
        # Summary section with individual breakdowns
        logger.warning("🎯 DETAILED BREAKDOWN BY COMPONENT:")
        logger.warning("")
        
        # Individual component breakdown
        for component_name, stats in sorted_stats:
            corr = stats['correlation']
            emoji = stats['emoji']
            influence_pct = (corr / total_correlation * 100) if total_correlation > 0 else 0
            logger.warning(f"   {emoji} {component_name:<20s}: {corr:.4f} correlation ({influence_pct:5.1f}% of total influence)")
            
        logger.warning("")
        logger.warning("📊 MODALITY GROUPINGS (for reference):")
        
        # Calculate grouped statistics for reference
        visual_components = ['drone_camera_vae', 'static_camera_vae']
        kinematic_components = ['drone_linear_vel', 'drone_angular_vel', 'drone_actions']
        spatial_components = ['drone_position', 'static_camera_pos', 'static_camera_orient', 'drone_orientation']
        
        visual_corr = sum(component_stats.get(comp, {}).get('correlation', 0) for comp in visual_components)
        kinematic_corr = sum(component_stats.get(comp, {}).get('correlation', 0) for comp in kinematic_components)
        spatial_corr = sum(component_stats.get(comp, {}).get('correlation', 0) for comp in spatial_components)
        
        if total_correlation > 0:
            visual_pct = (visual_corr / total_correlation) * 100
            kinematic_pct = (kinematic_corr / total_correlation) * 100
            spatial_pct = (spatial_corr / total_correlation) * 100
            
            logger.warning(f"   📹 Visual Information:   {visual_pct:5.1f}% ({visual_corr:.3f} total correlation)")
            logger.warning(f"      └─ 📷 Drone Camera:    {component_stats.get('drone_camera_vae', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ 📹 Static Camera:   {component_stats.get('static_camera_vae', {}).get('correlation', 0):.4f}")
            logger.warning("")
            logger.warning(f"   ⚡ Kinematic State:      {kinematic_pct:5.1f}% ({kinematic_corr:.3f} total correlation)")
            logger.warning(f"      └─ ⚡ Linear Velocity:  {component_stats.get('drone_linear_vel', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ 🌀 Angular Velocity: {component_stats.get('drone_angular_vel', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ 🎮 Action History:   {component_stats.get('drone_actions', {}).get('correlation', 0):.4f}")
            logger.warning("")
            logger.warning(f"   🧭 Spatial Awareness:    {spatial_pct:5.1f}% ({spatial_corr:.3f} total correlation)")
            logger.warning(f"      └─ 🎯 Drone Position:   {component_stats.get('drone_position', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ 📍 Static Cam Pos:   {component_stats.get('static_camera_pos', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ 🧭 Static Cam Orient:{component_stats.get('static_camera_orient', {}).get('correlation', 0):.4f}")
            logger.warning(f"      └─ ✈️ Drone Orientation:{component_stats.get('drone_orientation', {}).get('correlation', 0):.4f}")
            
        logger.warning("")
        logger.warning("💡 OPTIMIZATION INSIGHTS:")
        
        # Find highest and lowest influence components
        highest = sorted_stats[0] if sorted_stats else None
        lowest = sorted_stats[-1] if sorted_stats else None
        
        if highest:
            h_name, h_stats = highest
            logger.warning(f"   🏆 Most influential: {h_stats['emoji']} {h_name} ({h_stats['correlation']:.4f} correlation)")
            
        if lowest:
            l_name, l_stats = lowest
            if l_stats['correlation'] < 0.05:
                logger.warning(f"   ⚠️  Least influential: {l_stats['emoji']} {l_name} ({l_stats['correlation']:.4f} correlation)")
                logger.warning(f"      💡 Consider removing low-influence components for efficiency")
        
        # Efficiency recommendations
        unused_components = [name for name, stats in component_stats.items() 
                           if stats['correlation'] < 0.05]
        
        if unused_components:
            logger.warning(f"   🚫 Minimal influence components ({len(unused_components)}): {', '.join(unused_components)}")
            total_unused_dims = sum(self.obs_components[comp][1] - self.obs_components[comp][0] for comp in unused_components)
            logger.warning(f"      💡 Could remove {total_unused_dims}D from observation space")
            
        critical_components = [name for name, stats in component_stats.items() 
                             if stats['correlation'] > 0.2]
        
        if critical_components:
            logger.warning(f"   🔥 Critical components ({len(critical_components)}): {', '.join(critical_components)}")
            logger.warning(f"      ✅ These are essential for decision-making")
            
        logger.warning("================================================================================")
        
    def cleanup(self) -> None:
        """Remove hooks and cleanup resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.warning("🧹 Complete observation tracker cleaned up")


def create_influence_tracker(model, config: dict[str, object]) -> Optional[CompleteObservationInfluenceTracker]:
    """
    Factory function to create complete observation influence tracker.
    
    Args:
        model: The neural network model to monitor
        config: Configuration dictionary with tracking parameters
        
    Returns:
        CompleteObservationInfluenceTracker instance or None if creation fails
    """
    try:
        tracker = CompleteObservationInfluenceTracker(model, config)
        if tracker.enabled:
            logger.warning("✅ Complete observation influence tracker created successfully")
            return tracker
        else:
            logger.warning("❌ Complete observation influence tracker creation failed")
            return None
    except Exception as e:
        logger.warning(f"❌ Error creating complete observation influence tracker: {e}")
        return None


# For backwards compatibility
INFLUENCE_MONITOR_AVAILABLE = True 