# Kerala-Dam-Water-Levels — Data Summary

**Source**: https://github.com/amith-vp/Kerala-Dam-Water-Levels  
**Fetched**: 2026-07-13  
**Data Format**: JSON  

---

## KSEB Dams (historic_data/)

These are hydroelectric dams managed by Kerala State Electricity Board. Data is scraped from the KSEB dam monitoring portal.

### Idukki Dam
- **Records**: 2,014
- **Date Range**: 13 Aug 2020 → 10 Jul 2026 (newest first)
- **Fields**: date, waterLevel, liveStorage, storagePercentage, inflow, powerHouseDischarge, spillwayRelease, totalOutflow, rainfall
- **Known Gaps**: Data starts Aug 2020. **No coverage for 2018 flood event.** Daily resolution; missing days have not yet been audited programmatically.

### Idamalayar Dam
- **Records**: 2,014
- **Date Range**: 13 Aug 2020 → 10 Jul 2026 (newest first)
- **Fields**: Same as Idukki
- **Known Gaps**: Same as Idukki — starts Aug 2020, **no 2018 coverage**.

### Other KSEB Dams (also downloaded)
All 16 KSEB dams in the repo have been downloaded for completeness:
Anathode, Anayirankal, Banasura Sagar, Chenkulam, Erattayar, Kakkayam, Kallar, Kallarkutty, Kundala, Mattupetty, Moozhiyar, Pamba, Pambla, Ponmudi, Poringalkuthu, Sholayar.

---

## Irrigation Dams (irrigation_historic_data/)

These are irrigation/barrage dams managed by the Kerala Irrigation Department. Data is scraped separately.

### Bhoothathankettu Barrage ⚠️
- **Records**: 154
- **Date Range**: 03 Jan 2026 → 12 Jul 2026 (newest first)
- **Fields**: date, waterLevel, liveStorage, storagePercentage, inflow, powerHouseDischarge, spillwayRelease, totalOutflow, outflow, rainfall, remarks
- **CRITICAL GAP**: Only ~6 months of data available. **Does NOT go back to 2020 like the KSEB dams, and has NO 2018 coverage.**
- **Note**: Bhoothathankettu is a barrage (run-of-river), not a storage reservoir. Its water levels depend on upstream releases from Idukki and Idamalayar. The "liveStorage" field is present but may have different semantics from large reservoirs.

### Malankara Dam (also Periyar basin, relevant)
- Downloaded and available in `irrigation_historic_data/Malankara.json`

### All other irrigation dams (20 total)
All irrigation dams in the repo have been downloaded for completeness.

---

## Live Snapshots (root-level files)

### live.json
- Current KSEB dam levels snapshot (single point-in-time)

### irrigation_live.json
- Current irrigation dam levels snapshot (single point-in-time)

---

## Critical Implications for AquaFlow-CL

1. **2018 Flood Benchmark**: This GitHub dataset **cannot** provide 2018 flood data. The CWC flood report (Task 2) is the primary source for 2018 data.
2. **Bhoothathankettu Coverage**: With only 154 records (Jan–Jul 2026), Bhoothathankettu data from this source is insufficient for model training. The KSEB dams (Idukki, Idamalayar) have ~6 years of daily data and are the backbone of the training set.
3. **Recommended Next Step**: Audit missing days in Idukki/Idamalayar by parsing all dates and checking for gaps. This can be done in a preprocessing notebook.
