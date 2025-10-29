# show_env.py
import sys, site, pkgutil
print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace('\\n',' '))
print("sys.path[0:5]:", sys.path[:5])
print("site-packages locations:", site.getsitepackages() if hasattr(site, 'getsitepackages') else site.getusersitepackages())
print("\nInstalled top-level packages (first 50):")
for i, m in enumerate(sorted([p.name for p in pkgutil.iter_modules()])):
    if i>49: break
    print(" ", m)
