
    def loss_density(self, outputs, targets, indices, num_boxes):
        """Compute the density map loss (MSE)"""
        assert 'pred_density' in outputs
        pred_density = outputs['pred_density'] # [B, 1, H, W]
        
        # Generate GT density maps on the fly
        # We need to know the feature map shape H, W
        # pred_density is [B, 1, H, W]
        
        with torch.no_grad():
            gt_density = DensityGuidedQueryInit.generate_gt_density_map(
                targets, 
                pred_density.shape, 
                sigma=1.0 # Could be a hyperparam or dynamic based on image size
            )
            
        loss_density = F.mse_loss(pred_density, gt_density)
        
        return {'loss_density': loss_density * self.density_loss_coef}

