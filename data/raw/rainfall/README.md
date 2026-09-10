# data/raw/rainfall/ — Source Documentation

**Last updated**: 2026-07-13

---

## IMD Gridded Rainfall (0.25° × 0.25°)

### Source
- **Provider**: India Meteorological Department (IMD), Pune
- **Portal**: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- **Citation**: Pai D.S., Latha Sridhar, Rajeevan M., Sreejith O.P., Satbhai N.S. and Mukhopadhyay B., 2014: *Development of a new high spatial resolution (0.25° X 0.25°) Long period (1901-2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region*; MAUSAM, 65, 1(January 2014), pp1-18.

### Data Specification
- **Spatial Resolution**: 0.25° × 0.25° (~25 km)
- **Temporal Resolution**: Daily
- **Coverage**: All-India (6.5°N–38.5°N, 66.5°E–100.0°E)
- **Grid Size**: 135 × 129 grid points
- **Unit**: Millimeters (mm)
- **Format**: NetCDF (.nc)
- **Available Range**: 1901–2025

### Files Downloaded

| File | Year | Size (MB) | Days | Purpose |
|------|------|-----------|------|---------|
| `IMD_RF25_2017.nc` | 2017 | 24.3 | 365 | Pre-flood baseline |
| `IMD_RF25_2018.nc` | 2018 | 24.3 | 365 | **2018 Kerala Flood Event** (critical) |
| `IMD_RF25_2019.nc` | 2019 | 24.3 | 365 | Post-flood comparison |
| `IMD_RF25_2023.nc` | 2023 | 24.3 | 365 | Recent — model training/validation |
| `IMD_RF25_2024.nc` | 2024 | 24.3 | 366 | Recent — model training/validation |
| `IMD_RF25_2025.nc` | 2025 | 24.3 | 365* | Recent — model training/validation |

*2025 data may be partial depending on when IMD updates the file.

### Last Fetched
- **Date**: 2026-07-13
- **Method**: Automated download via POST to `RF25.php` on IMD portal

---

## Grid Coordinates for Periyar Basin

The three target reservoirs and their approximate grid coordinates in the IMD dataset:

| Reservoir | Latitude (°N) | Longitude (°E) | IMD Grid Index (approx) |
|-----------|--------------|----------------|------------------------|
| Idukki | 9.84 | 76.97 | J≈14, I≈42 |
| Idamalayar | 10.22 | 76.72 | J≈15, I≈41 |
| Bhoothathankettu | 10.13 | 76.63 | J≈15, I≈41 |

**Periyar Basin Bounding Box** (for slicing the NetCDF to area of interest):
- **Latitude**: 9.5°N to 10.5°N
- **Longitude**: 76.5°E to 77.5°E

Grid indices (0-based):
- **J (lat)**: index 12 to 16 → (9.5 = 6.5 + 12×0.25) to (10.5 = 6.5 + 16×0.25)
- **I (lon)**: index 40 to 44 → (76.5 = 66.5 + 40×0.25) to (77.5 = 66.5 + 44×0.25)

This gives a 5×5 grid (25 cells) covering the entire Periyar catchment.

---

## Known Caveats
- ⚠️ The NetCDF files cover all of India. You must **slice to the Periyar basin subset** during preprocessing.
- ⚠️ The grid uses gauge-interpolated data, not direct satellite estimates. Coastal/mountainous areas (like the Western Ghats) may have lower gauge density.
- ⚠️ 2025 data file may not contain full 365 days if downloaded mid-year.
- The data is arranged South-to-North (J=1 at 6.5°N) and West-to-East (I=1 at 66.5°E).
