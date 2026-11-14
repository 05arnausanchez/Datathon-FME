import zipfile
import os
from pathlib import Path

zip_path = Path('submission.zip')
# Estructura plana: agent.py y modelos a la raíz del ZIP (según instrucciones del datathon)
files = [
    'agent.py',
    'models/model_price.pkl',
    'models/model_co2.pkl',
    'models/model_pv.pkl',
    'models/feature_cols.pkl',
]

# Eliminar ZIP anterior si existe
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = Path(f)
        if p.exists():
            # Guardar a la raíz del ZIP (solo el nombre del archivo, sin carpeta 'models/')
            arcname = p.name
            z.write(f, arcname)
            print(f'  Agregado: {arcname} ({os.path.getsize(f)/1e6:.1f} MB)')
        else:
            print(f'  ADVERTENCIA: {f} no existe, omitido')

print(f'\nsubmission.zip creado: {os.path.getsize(zip_path)/1e6:.1f} MB')
print('Estructura (plana, como exige el datathon):')
with zipfile.ZipFile(zip_path, 'r') as z:
    for info in z.infolist():
        print(f'  {info.filename} ({info.file_size/1e6:.1f} MB)')