# LigaMX Stats

Análisis estadístico de la Liga MX usando datos de FotMob.

## Estructura del proyecto

```
LigaMX_Stats/
├── data/
│   ├── raw/          # Datos crudos descargados de FotMob (JSONs, CSVs)
│   └── processed/    # Datos limpios y listos para análisis
├── notebooks/        # Jupyter notebooks de exploración y análisis
├── scripts/          # Scripts .py reutilizables
├── output/
│   ├── charts/       # Gráficas exportadas (PNGs)
│   └── reports/      # Reportes finales
└── README.md
```

## Setup

```bash
# Activar entorno virtual
source .venv/bin/activate

# Iniciar Jupyter
jupyter lab
```

## Librerías principales

- [fotmob-api](https://github.com/bgjoseluis/fotmob-api) — extracción de datos de FotMob
- [mplsoccer](https://mplsoccer.readthedocs.io/) — visualizaciones de fútbol
- [pandas](https://pandas.pydata.org/) — manipulación de datos
- [soccerdata](https://soccerdata.readthedocs.io/) — datos adicionales de fútbol
