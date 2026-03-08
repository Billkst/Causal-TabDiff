import re

with open("src/baselines/wrappers.py", "r") as f:
    text = f.read()

# Make sure we don't apply it multiple times
if "best_loss = float('inf')" not in text:
    old_code = """        for epoch in range(epochs):
            num_batches = len(dataloader) if not debug_mode else min(2, len(dataloader))
            epoch_loss = 0
            log_every = 20"""

    new_code = """        # Early Stopping Setup
        patience = 10
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # When using early stopping, we can allow a higher upper bound
        max_epochs = 300 if not debug_mode else epochs
        
        for epoch in range(max_epochs):
            num_batches = len(dataloader) if not debug_mode else min(2, len(dataloader))
            epoch_loss = 0
            epoch_diff_loss = 0
            log_every = 20"""

    text = text.replace(old_code, new_code)
    
    old_loss_acc = """                diff_loss, disc_loss, outcome_loss = self.model(
                    x,
                    alpha_tgt,
                    y,
                    pos_weight=pos_weight
                )
                
                loss = diff_loss + disc_loss + outcome_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                if (i+1) % log_every == 0 or i == num_batches - 1:"""
    
    new_loss_acc = """                diff_loss, disc_loss, outcome_loss = self.model(
                    x,
                    alpha_tgt,
                    y,
                    pos_weight=pos_weight
                )
                
                loss = diff_loss + disc_loss + outcome_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_diff_loss += diff_loss.item()
                
                if (i+1) % log_every == 0 or i == num_batches - 1:"""
    
    text = text.replace(old_loss_acc, new_loss_acc)

    old_epoch_end = """            # Outcome pseudo-labeling trainer hook
            if epoch == 0 and len(outcome_features) > 0:"""
            
    new_epoch_end = """            avg_epoch_loss = epoch_loss / num_batches
            avg_diff_loss = epoch_diff_loss / num_batches
            logger.info(f"[Causal-TabDiff] Epoch {epoch + 1}/{max_epochs} Complete - Avg Loss: {avg_epoch_loss:.4f} (Diff: {avg_diff_loss:.4f})")
            
            # Check Early Stopping based on diff_loss (or total loss)
            if avg_diff_loss < best_loss - 1e-4:
                best_loss = avg_diff_loss
                patience_counter = 0
                import copy
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                logger.info(f"[Causal-TabDiff] Early Stopping counter: {patience_counter}/{patience} (Best Diff: {best_loss:.4f})")
                
            if patience_counter >= patience:
                logger.info(f"[Causal-TabDiff] Early Stop triggered at epoch {epoch + 1}! Restoring best model state.")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break

            # Outcome pseudo-labeling trainer hook
            if epoch == 0 and len(outcome_features) > 0:"""
            
    text = text.replace(old_epoch_end, new_epoch_end)

    with open("src/baselines/wrappers.py", "w") as f:
        f.write(text)
    print("Early stopping injected.")
