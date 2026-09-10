# data/raw/reservoir/ — Source Documentation

**Last updated**: 2026-07-13

---

## Folder Structure

### Kerala-Dam-Water-Levels/
- **Source**: https://github.com/amith-vp/Kerala-Dam-Water-Levels
- **Type**: JSON files scraped from KSEB and Kerala Irrigation Department portals
- **Date Range**: Aug 2020 → Jul 2026 (KSEB dams), Jan 2026 → Jul 2026 (Irrigation dams)
- **Last Fetched**: 2026-07-13
- **Contents**:
  - `live.json` — Latest snapshot of all KSEB dam levels
  - `irrigation_live.json` — Latest snapshot of all irrigation dam levels
  - `historic_data/` — Daily time-series for 18 KSEB hydroelectric dams (Idukki, Idamalayar, Sholayar, etc.)
  - `irrigation_historic_data/` — Daily time-series for 20 irrigation dams (Bhoothathankettu, Malankara, etc.)
  - `data_summary.md` — Detailed record counts, date ranges, and gap analysis per dam

#### Key Dams for AquaFlow-CL (Periyar Basin)
| Dam | Type | Source Folder | Records | Date Range | 2018 Data? |
|-----|------|---------------|---------|------------|------------|
| Idukki | KSEB (Hydro) | `historic_data/Idukki.json` | 2,014 | Aug 2020 – Jul 2026 | ❌ No |
| Idamalayar | KSEB (Hydro) | `historic_data/Idamalayar.json` | 2,014 | Aug 2020 – Jul 2026 | ❌ No |
| Bhoothathankettu | Irrigation (Barrage) | `irrigation_historic_data/Bhoothathankettu_(Barrage).json` | 154 | Jan 2026 – Jul 2026 | ❌ No |

#### Known Gaps & Caveats
- ⚠️ **GitHub scraper data starts Aug 2020, NOT usable for 2018 flood baseline.**
- ⚠️ Bhoothathankettu has only ~6 months of data (far less than KSEB dams).
- The data fields include: date, waterLevel, liveStorage, storagePercentage, inflow, powerHouseDischarge, spillwayRelease, totalOutflow, rainfall.

---

### CWC/
- **Source**: https://sdma.kerala.gov.in/wp-content/uploads/2020/08/CWC-Report-on-Kerala-Floods.pdf
- **Type**: PDF report + partial manual CSV extraction
- **Date Range**: June – August 2018 (2018 flood event analysis)
- **Last Fetched**: 2026-07-13
- **Contents**:
  - `CWC-Report-on-Kerala-Floods.pdf` — Full 60+ page CWC study report on Kerala Floods 2018
  - `idukki_idamalayar_2018_manual_extract.csv` — Partial extraction of timestamped numeric data points from the report text
  - `EXTRACTION_STATUS.md` — Detailed guide on what was extracted and what requires manual work (Fig.4, Fig.5, Tables 12-13)

#### Known Gaps & Caveats
- ⚠️ Fig.4 (Idukki) and Fig.5 (Idamalayar) are embedded as raster chart images — need manual digitization via WebPlotDigitizer.
- ⚠️ Tables 12-13 (Sholayar/Parambikulam data) are in the PDF but not yet extracted to CSV. Recommend `pdfplumber` in Month 2.
- ⚠️ The manual CSV extract is **incomplete** — only contains well-cited data points from academic literature.

---

### KSEB/
- **Status**: Empty (placeholder)
- **Intended Use**: If KSEB operational logs or generation data (separate from the GitHub scraper) are obtained via direct request, they go here.

---

### Other/
- **Source**: Web searches and Wayback Machine archives
- **Contents**:
  - `bhoothathankettu_2018_GAP.md` — Detailed report confirming that no publicly available daily time-series exists for Bhoothathankettu during Aug 2018, with full audit trail of all sources searched.

#### Known Gaps & Caveats
- ⚠️ Bhoothathankettu 2018 daily data is a **confirmed gap**. Recommended to derive from upstream Idukki + Idamalayar releases or file an RTI request.
