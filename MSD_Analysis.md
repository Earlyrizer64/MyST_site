---
title: MSD Analysis
kernelspec:
  name: python3
  display_name: Python 3 (Pyodide)
---

# MSD Analysis

```{code-cell} python
import micropip
await micropip.install("ipywidgets")
await micropip.install("matplotlib")

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import io

upload = widgets.FileUpload(accept='.dat', multiple=False)
display(upload)
print("Upload your MSD .dat file above, then run the next cell.")
```

```{code-cell} python
content = upload.value[0]['content']
text = io.BytesIO(bytes(content)).read().decode('utf-8')

# Parse — skip comment lines starting with #
lines = [l for l in text.strip().split('\n') if not l.startswith('#')]
data = np.array([list(map(float, l.split())) for l in lines if l.strip()])

timesteps = data[:, 0]
msd = data[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(timesteps, msd, color='#185FA5', linewidth=1.5)
plt.xlabel('Timestep')
plt.ylabel('MSD (Å²)')
plt.title('Mean Squared Displacement')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

slope = np.polyfit(timesteps[len(timesteps)//4:], msd[len(msd)//4:], 1)[0]
print(f"Estimated diffusion coefficient D ≈ {slope/6:.4e} Å²/timestep")
```
