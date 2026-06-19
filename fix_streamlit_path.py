import winreg
from pathlib import Path

script_dir = Path(r"C:\Program Files\Python314\Scripts")
python_dir = Path(r"C:\Program Files\Python314")

print("Script folder exists:", script_dir.exists())
print("streamlit.exe exists:", (script_dir / "streamlit.exe").exists())
print("uvicorn.exe exists:", (script_dir / "uvicorn.exe").exists())

try:
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ)
    current_path = winreg.QueryValueEx(key, "Path")[0]
    winreg.CloseKey(key)
except FileNotFoundError:
    current_path = ""

entries = [p for p in current_path.split(";") if p]
changed = False
for candidate in (str(script_dir), str(python_dir)):
    if candidate not in entries:
        entries.append(candidate)
        changed = True

if changed:
    new_path = ";".join(entries)
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, "Environment")
    winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
    winreg.CloseKey(key)
    print("Updated user PATH with:")
    print(str(script_dir))
    print(str(python_dir))
else:
    print("User PATH already contains the required entries.")

print("Done. Restart PowerShell to use the updated PATH.")
