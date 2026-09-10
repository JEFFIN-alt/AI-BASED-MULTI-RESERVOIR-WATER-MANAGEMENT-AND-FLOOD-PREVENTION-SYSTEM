# Bhoothathankettu Dam — August 2018 Data Gap Report

**Date**: 2026-07-13  
**Searched by**: Automated data acquisition pipeline  
**Status**: ❌ NO RELIABLE DAILY DATASET FOUND

---

## What Was Searched

### 1. GitHub: amith-vp/Kerala-Dam-Water-Levels
- **Result**: `irrigation_historic_data/Bhoothathankettu_(Barrage).json` exists but only contains **154 records from Jan 2026 → Jul 2026**. No 2018 or pre-2026 data.
- The KSEB `historic_data/` folder does not include Bhoothathankettu (it's an irrigation barrage, not a KSEB power station).

### 2. KSDMA Dam Water Level Portal via Wayback Machine
- **URL checked**: `https://web.archive.org/web/20180815/https://sdma.kerala.gov.in/dam-water-level/`
- **Result**: The archived page snapshot is from late 2019/early 2020. It lists daily PDF bulletins but only for Dec 2019 / Jan 2020 dates. No August 2018 bulletins are indexed in the Wayback Machine captures for this page.
- The KSDMA portal appears to have published daily PDFs (KSEB status + Irrigation status), but these historical PDFs from August 2018 are not archived.

### 3. Web Search: News Reports
- **Result**: Scattered individual data points from news articles (e.g., New Indian Express):
  - **Aug 10, 2018 at 7:00 PM**: Water level = 30.85 m, discharge ≈ 7,079 cumec, all 15 shutters open.
  - **Peak inflow estimate**: ~7,700 m³/s during the flood peak.
- These are isolated snapshots, NOT a continuous daily time-series.

### 4. Web Search: Kerala Irrigation Department Archives
- **Result**: No public CSV/API endpoint found for historical daily dam data. The search confirmed that daily records may only be available via a formal RTI (Right to Information) request to the Kerala Irrigation Department.

### 5. India-WRIS / CWC Data Portal
- **Result**: The CWC flood report focuses on Idukki and Idamalayar. Bhoothathankettu is mentioned in the context of downstream impact but does NOT have its own dedicated figure or data table in the report.

---

## Confirmed Gap

**There is no publicly available, machine-readable daily time-series dataset for Bhoothathankettu dam water levels during August 2018.**

The only data points available are:
- Scattered news reports with individual timestamps
- The CWC report's qualitative descriptions of downstream conditions

---

## Recommended Actions

1. **Derive from upstream**: Since Bhoothathankettu is a run-of-river barrage on the Periyar, its inflow is largely the sum of releases from Idukki (via Cheruthoni) and Idamalayar, plus local catchment runoff. The 2018 data for Idukki and Idamalayar (from the CWC report) can be used to **estimate** Bhoothathankettu inflow with a time lag and a free-catchment correction factor.

2. **RTI Request**: File an RTI application with the Executive Engineer, Irrigation Design & Research Bureau, Thiruvananthapuram, requesting daily operational logs for Bhoothathankettu Barrage for June–September 2018.

3. **Model as derived variable**: In the AquaFlow-CL system, treat Bhoothathankettu not as an independent reservoir but as a **downstream aggregation node** whose state is computed from upstream releases. This is physically accurate since it is a barrage (minimal storage), not a large reservoir.

---

## Sources Consulted
- https://github.com/amith-vp/Kerala-Dam-Water-Levels
- https://web.archive.org/web/20180815/https://sdma.kerala.gov.in/dam-water-level/
- https://sdma.kerala.gov.in/wp-content/uploads/2020/08/CWC-Report-on-Kerala-Floods.pdf
- New Indian Express, August 2018 flood coverage
- India-WRIS (https://india-wris.gov.in/)
