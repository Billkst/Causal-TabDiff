import sys

with open("src/baselines/wrappers.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "epoch_loss += loss.item()" in line:
        if "epoch_diff_loss += diff_loss.item()" not in lines[i+1]:
            new_lines.append("                epoch_diff_loss += diff_loss.item()\n")
            
    if "logger.info(f\"[Causal-TabDiff] Epoch {epoch + 1}/{epochs} - avg_loss={avg_loss:.6f}, batches={num_batches}\")" in line:
        new_lines.append("""
            avg_epoch_loss = epoch_loss / num_batches
            avg_diff_loss = epoch_diff_loss / num_batches
            logger.info(f"[Causal-TabDiff] Epoch {epoch + 1}/{max_epochs} Complete - Avg Loss: {avg_epoch_loss:.4f} (Diff: {avg_diff_loss:.4f})")
            
            # Check Early Stopping based on diff_loss
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
""")

with open("src/baselines/wrappers.py", "w") as f:
    f.writelines(new_lines)
