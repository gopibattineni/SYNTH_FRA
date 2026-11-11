"""Write multi-seed results as Excel workbooks (stdlib, no openpyxl required)."""
from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

NUMERIC_HEADERS = {
    "seed",
    "seed_55",
    "seed_155",
    "seed_255",
    "seed_355",
    "mean",
    "sd",
    "n_seeds",
    "n_synthetic_samples",
    "n_train",
    "n_test",
    "split_seed",
    "metric_value",
    "training_time_seconds",
    "generation_time_seconds",
    "evaluation_time_seconds",
    "total_time_seconds",
}
INT_HEADERS = {"seed", "n_seeds", "n_synthetic_samples", "n_train", "n_test", "split_seed"}
CATEGORY_FILES = [
    ("Privacy", "privacy_mean_sd.csv"),
    ("Utility", "utility_mean_sd.csv"),
    ("Fidelity", "fidelity_mean_sd.csv"),
    ("Compute", "compute_mean_sd.csv"),
    ("All metrics", "all_metrics_mean_sd.csv"),
]


def col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _xml_text(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch == "\t" or ch == "\n" or ord(ch) >= 32)
    return escape(cleaned)


def _finite_number(value):
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return value
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _parse_cell(header: str, raw: str):
    if raw is None or raw == "":
        return None, None
    text = str(raw).strip()
    if text.lower() in {"nan", "none", "inf", "-inf"}:
        return None, None
    # Wide "mean±SD" cells must stay text even when the column is a numeric metric
    # name such as evaluation_time_seconds.
    if "±" in text or "(n=" in text:
        return str(raw), "s"
    if header in NUMERIC_HEADERS:
        num = _finite_number(raw)
        if num is None:
            return None, None
        if header in INT_HEADERS:
            return int(num), "n"
        return num, "n"
    return raw, "s"


def sheet_xml(rows: list[list]) -> str:
    if not rows:
        rows = [[""]]
    headers = [str(h) for h in rows[0]]
    n_cols = len(headers)
    n_rows = len(rows)
    last = f"{col_letter(n_cols)}{n_rows}"
    widths = []
    for c, header in enumerate(headers):
        max_len = len(header)
        for row in rows[1:]:
            val = row[c] if c < len(row) else ""
            max_len = max(max_len, len(str(val if val is not None else "")))
        width = min(max(12, max_len + 2), 36)
        if header == "mean_sd":
            width = 22
        widths.append(width)
    cols_xml = "".join(
        f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
        for i, w in enumerate(widths, 1)
    )
    row_xml = []
    for r_idx, row in enumerate(rows, 1):
        cells = []
        for c_idx, header in enumerate(headers, 1):
            raw = row[c_idx - 1] if c_idx - 1 < len(row) else ""
            if r_idx == 1:
                # Header labels must stay text. Names like "mean"/"sd"/"seed_55"
                # are also numeric column types and must not be parsed as numbers.
                value, kind = ("" if raw is None else str(raw)), "s"
                if value == "":
                    value, kind = None, None
            elif not isinstance(raw, str) and raw is not None:
                if isinstance(raw, float):
                    value, kind = (None, None) if not math.isfinite(raw) else (raw, "n")
                elif isinstance(raw, int) and not isinstance(raw, bool):
                    value, kind = raw, "n"
                else:
                    value, kind = _parse_cell(header, str(raw))
            else:
                value, kind = _parse_cell(header, raw)
            ref = f"{col_letter(c_idx)}{r_idx}"
            style = 1 if r_idx == 1 else (2 if r_idx % 2 == 0 else 0)
            style_attr = f' s="{style}"' if style else ""
            if value is None:
                cells.append(f'<c r="{ref}"{style_attr}/>')
            elif kind == "n":
                cells.append(f'<c r="{ref}"{style_attr} t="n"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}"{style_attr} t="inlineStr"><is>'
                    f'<t xml:space="preserve">{_xml_text(value)}</t></is></c>'
                )
        row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
           xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetPr>
    <pageSetUpPr fitToPage="true"/>
  </sheetPr>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <cols>{cols_xml}</cols>
  <sheetData>{"".join(row_xml)}</sheetData>
  <autoFilter ref="A1:{last}"/>
  <pageSetup orientation="landscape" fitToWidth="1" fitToHeight="0"/>
</worksheet>'''


CONTENT_TYPES_TMPL = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  {overrides}
</Types>'''

RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E79"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD6EAF8"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFBFBFBF"/></left>
      <right style="thin"><color rgb="FFBFBFBF"/></right>
      <top style="thin"><color rgb="FFBFBFBF"/></top>
      <bottom style="thin"><color rgb="FFBFBFBF"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="0" fontId="0" fillId="3" borderId="1" xfId="0" applyFill="1" applyBorder="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>'''


def _safe_sheet_name(name: str) -> str:
    cleaned = "".join("_" if ch in r":\/?*[]" else ch for ch in name).strip() or "Sheet"
    return cleaned[:31]


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(_safe_sheet_name(name))}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{sheets}</sheets>
</workbook>'''


def workbook_rels(n_sheets: int) -> str:
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n_sheets + 1)
    ]
    rels.append(
        f'<Relationship Id="rId{n_sheets + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{"".join(rels)}
</Relationships>'''


def write_xlsx(path: Path, sheets: list[tuple[str, list[list]]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sheets:
        sheets = [("Sheet1", [["empty"]])]
    overrides = "\n  ".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, len(sheets) + 1)
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES_TMPL.format(overrides=overrides))
        zf.writestr("_rels/.rels", RELS)
        zf.writestr("xl/workbook.xml", workbook_xml([name for name, _ in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels(len(sheets)))
        zf.writestr("xl/styles.xml", STYLES)
        for i, (_, rows) in enumerate(sheets, 1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(rows))
    return path


def csv_to_rows(path: Path) -> list[list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def records_to_rows(records) -> list[list]:
    if hasattr(records, "to_dict"):
        records = records.where(records.notna(), None).to_dict("records")
    records = list(records)
    if not records:
        return []
    headers = list(records[0].keys())
    rows = [headers]
    for rec in records:
        rows.append([None if rec.get(h) is None else rec.get(h) for h in headers])
    return rows


def _wide_mean_sd(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return [["dataset", "generator"]]
    headers = rows[0]
    idx = {h: i for i, h in enumerate(headers)}
    needed = ("dataset", "generator", "metric_name", "mean_sd")
    if any(k not in idx for k in needed):
        return rows
    metrics = []
    seen = set()
    keys = []
    key_seen = set()
    lookup = {}
    for row in rows[1:]:
        metric = row[idx["metric_name"]] if idx["metric_name"] < len(row) else ""
        dataset = row[idx["dataset"]] if idx["dataset"] < len(row) else ""
        generator = row[idx["generator"]] if idx["generator"] < len(row) else ""
        mean_sd = row[idx["mean_sd"]] if idx["mean_sd"] < len(row) else ""
        if metric not in seen:
            seen.add(metric)
            metrics.append(metric)
        key = (dataset, generator)
        if key not in key_seen:
            key_seen.add(key)
            keys.append(key)
        lookup[(dataset, generator, metric)] = mean_sd
    out = [["dataset", "generator", *metrics]]
    for dataset, generator in keys:
        out.append(
            [dataset, generator, *[lookup.get((dataset, generator, m), "") for m in metrics]]
        )
    return out


def write_aggregated_excel(agg_dir: Path, records=None) -> list[Path]:
    """Write individual category workbooks plus one combined easy-read workbook."""
    agg_dir = Path(agg_dir)
    written: list[Path] = []
    combined: list[tuple[str, list[list]]] = []
    for sheet_name, csv_name in CATEGORY_FILES:
        csv_path = agg_dir / csv_name
        if records is not None and csv_name == "all_metrics_mean_sd.csv":
            rows = records_to_rows(records)
        elif records is not None:
            cat = sheet_name if sheet_name != "All metrics" else None
            if cat:
                recs = records
                if hasattr(recs, "to_dict"):
                    recs = recs[recs["metric_category"] == cat]
                else:
                    recs = [r for r in recs if r.get("metric_category") == cat]
                rows = records_to_rows(recs)
            else:
                rows = records_to_rows(records)
        elif csv_path.exists():
            rows = csv_to_rows(csv_path)
        else:
            continue
        if not rows:
            continue
        xlsx_path = agg_dir / csv_name.replace(".csv", ".xlsx")
        write_xlsx(xlsx_path, [(sheet_name, rows)])
        written.append(xlsx_path)
        if sheet_name != "All metrics":
            combined.append((f"{sheet_name} mean±SD", _wide_mean_sd(rows)))
        combined.append((sheet_name, rows))
    if combined:
        written.append(write_xlsx(agg_dir / "multi_seed_results.xlsx", combined))
    return written


def write_csv_as_xlsx(csv_path: Path, sheet_name: str | None = None) -> Path:
    csv_path = Path(csv_path)
    rows = csv_to_rows(csv_path)
    name = sheet_name or csv_path.stem.replace("_", " ")[:31]
    out = csv_path.with_suffix(".xlsx")
    write_xlsx(out, [(name, rows)])
    return out


def export_task_results(task_root: Path) -> list[Path]:
    """Convert aggregated tables, experiment log, and combined raw results."""
    task_root = Path(task_root)
    written: list[Path] = []
    agg = task_root / "results" / "aggregated"
    if agg.exists():
        written.extend(write_aggregated_excel(agg))
    log_csv = task_root / "results" / "logs" / "experiment_log.csv"
    if log_csv.exists():
        written.append(write_csv_as_xlsx(log_csv, "Experiment log"))
    raw_csv = task_root / "results" / "raw" / "multi_seed_raw_results.csv"
    if raw_csv.exists():
        written.append(write_csv_as_xlsx(raw_csv, "Raw results"))
    return written
