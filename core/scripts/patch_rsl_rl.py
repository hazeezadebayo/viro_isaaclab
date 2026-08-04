import os

target_path = "/isaac-sim/kit/python/lib/python3.10/site-packages/rsl_rl/modules/actor_critic.py"
if os.path.exists(target_path):
    with open(target_path, "r") as f:
        content = f.read()
    content = content.replace("std = torch.exp(self.log_std).expand_as(mean)", "std = torch.nan_to_num(torch.exp(self.log_std), nan=0.1).clamp(min=1e-3, max=10.0).expand_as(mean)")
    content = content.replace("mean = torch.nan_to_num(self.actor(observations), nan=0.0)", "mean = torch.nan_to_num(self.actor(observations), nan=0.0).clamp(min=-100.0, max=100.0)")
    content = content.replace("mean = self.actor(observations)", "mean = torch.nan_to_num(self.actor(observations), nan=0.0).clamp(min=-100.0, max=100.0)")
    content = content.replace("std = torch.nan_to_num(self.std, nan=0.1).clamp(min=1e-3).expand_as(mean)", "std = torch.nan_to_num(self.std, nan=0.1).clamp(min=1e-3, max=10.0).expand_as(mean)")
    content = content.replace("std = self.std.expand_as(mean)", "std = torch.nan_to_num(self.std, nan=0.1).clamp(min=1e-3, max=10.0).expand_as(mean)")
    with open(target_path, "w") as f:
        f.write(content)
    print("[OK] Patched rsl_rl actor_critic.py with full std & log_std bounds")
