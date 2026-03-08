import re
with open("src/baselines/wrappers.py", "r") as f:
    text = f.read()

text = re.sub(r'x_cf_semantic = self._decode_semantic_from_analog\(sampled, device\).*?\n\s+glue_score = self._predict_outcome_score_from_semantic', 
r'x_cf_semantic = self._decode_semantic_from_analog(sampled, device)\n        x_cf_semantic = torch.clamp(x_cf_semantic, min=-5.0, max=5.0)\n        glue_score = self._predict_outcome_score_from_semantic', text, flags=re.DOTALL)

with open("src/baselines/wrappers.py", "w") as f:
    f.write(text)
print("Regex replaced")
