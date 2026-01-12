Opción A: ejecutar en local “tal cual” (recomendado)
1.	Crea un entorno (opcional pero aconsejable):
    python -m venv .venv
    # mac/linux
    source .venv/bin/activate
    # windows
    .venv\Scripts\activate
2.	Instala deps:
    pip install fastapi uvicorn pandas
3.	Lanza el servidor:
    uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
    En desarrollo, con autoreload:
    uvicorn api_fastapi:app --host 127.0.0.1 --port 8000 --reload
4.	Abre:
	•	Swagger: http://127.0.0.1:8000/docs
	•	Health: http://127.0.0.1:8000/health
Opción B: que tu front (otro dispositivo) lo consuma en tu red
    Si tu front corre en otro PC/móvil de la misma red, expón en LAN:
    uvicorn api_fastapi:app --host 0.0.0.0 --port 8000
    Y accedes desde el front a:
	•	http://IP_DE_TU_PC:8000/...

    Tip: en algunos firewalls tendrás que permitir el puerto 8000.

    Apuntar a tus rutas reales (si no están en la carpeta esperada)

Antes de arrancar, define variables:
    # mac/linux
    export OMDB_CACHE_PATH="/ruta/a/omdb_cache.json"
    export WIKI_CACHE_PATH="/ruta/a/wiki_cache.json"
    export REPORT_ALL_PATH="/ruta/a/report_all.csv"
    export REPORT_FILTERED_PATH="/ruta/a/report_filtered.csv"
    export METADATA_FIX_PATH="/ruta/a/metadata_fix.csv"

En Windows (PowerShell):
    $env:OMDB_CACHE_PATH="C:\ruta\omdb_cache.json"
    $env:WIKI_CACHE_PATH="C:\ruta\wiki_cache.json"
    $env:REPORT_ALL_PATH="C:\ruta\report_all.csv"
    $env:REPORT_FILTERED_PATH="C:\ruta\report_filtered.csv"
    $env:METADATA_FIX_PATH="C:\ruta\metadata_fix.csv"

    CORS para tu front local (Vite/React/Next)

Si tu front está en http://localhost:5173, puedes restringir CORS así:
    # mac/linux
    export CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
    uvicorn api_fastapi:app --host 127.0.0.1 --port 8000
    (En dev también puedes dejarlo abierto, pero mejor restringir si ya sabes el origen.)

    Ejemplos rápidos de consumo desde el front
    Listar report_all paginado
    fetch("http://127.0.0.1:8000/reports/all?limit=50&offset=0&query=matrix")
    .then(r => r.json())
    .then(console.log);
    OMDb por IMDb ID
    fetch("http://127.0.0.1:8000/cache/omdb/by-imdb/tt0133093")
    .then(r => r.json())
    .then(console.log);
Bonus (muy útil): comprobar qué ficheros está viendo la API
    Llama:
        •	GET /meta/files

    Ahí verás exactamente si existen, tamaño, mtime y hash.

3️⃣ Importar en Postman
	1.	Abre Postman
	2.	Clic en Import
	3.	Elige una de estas opciones:
	•	Link → pega http://localhost:8000/openapi.json
	•	File → descarga primero el openapi.json y súbelo
	4.	Pulsa Import

✅ Postman creará automáticamente:
	•	Una Collection
	•	Todas las rutas
	•	Métodos (GET, POST, etc.)
	•	Parámetros
	•	Esquemas de request/response