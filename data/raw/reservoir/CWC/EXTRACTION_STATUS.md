# CWC Report Data — Extraction Status & Manual Steps Required

**Source PDF**: `CWC-Report-on-Kerala-Floods.pdf` (6 MB, 60+ pages)  
**Downloaded**: 2026-07-13 from https://sdma.kerala.gov.in/wp-content/uploads/2020/08/CWC-Report-on-Kerala-Floods.pdf  

---

## What Was Extracted Automatically

### idukki_idamalayar_2018_manual_extract.csv
A **partial** CSV containing specific numeric data points that are widely cited in academic literature about the CWC report. These are approximate values from narrative text — NOT digitized from the figures.

**⚠️ THIS FILE IS INCOMPLETE. Manual extraction is required.**

---

## What Requires Manual Extraction

### 1. Fig.4 — Idukki Reservoir (Inflow/Outflow/Water Level Chart)
- **Location in PDF**: Approximately pages 28–30
- **Action Needed**: 
  - Screenshot/export the figure as `idukki_fig4_inflow_outflow.png`
  - Digitize the chart using WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/) to extract daily or sub-daily values into a CSV
  - Expected columns: `datetime, water_level_m, inflow_cumec, outflow_cumec`

### 2. Fig.5 — Idamalayar Reservoir (Inflow/Outflow/Water Level Chart)
- **Location in PDF**: Approximately pages 30–32
- **Action Needed**: Same as Fig.4
  - Save as `idamalayar_fig5_inflow_outflow.png`
  - Digitize with WebPlotDigitizer

### 3. Table 12 — Kerala Sholayar Daily Inflow-Spill Data
- **Location in PDF**: Approximately pages 35–40
- **Action Needed**: Copy-paste the table into a spreadsheet and export as CSV
  - Expected columns: `date, inflow_cumec, spill_cumec`

### 4. Table 13 — Parambikulam / Tunakadavu Daily Data
- **Location in PDF**: Near Table 12
- **Action Needed**: Same as Table 12

### 5. All Timestamped Numeric Data in Narrative Text
- **Action Needed**: Read through the full report and extract every line like:
  - "731.82 m at 00:00 hrs on 10 August 2018"
  - "peak inflow of 2500 cumec on 15 August"
- Add to `idukki_idamalayar_2018_manual_extract.csv` with the page number as source

---

## Recommended Tool for Figure Digitization

**WebPlotDigitizer** (https://automeris.io/WebPlotDigitizer/)
- Free, browser-based
- Upload chart screenshot → click data points → export CSV
- This is the standard method used in hydrology research when only figures are available

---

## Why Automated Extraction Failed

The CWC report is a scanned/image-heavy PDF. The key data (Fig.4, Fig.5) are embedded as raster images (charts), not as text tables. Standard PDF text extraction tools cannot parse chart images. Python libraries like `tabula-py` or `pdfplumber` could extract the text tables (12, 13) but are not yet installed in this project environment.

**Recommended**: Install `pdfplumber` later (add to requirements.txt during Month 2) and write a notebook to extract Tables 12–13 programmatically.
