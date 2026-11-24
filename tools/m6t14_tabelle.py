#!/usr/bin/env python3
"""Vergleichstabelle für M6-T14 (Bericht docs/ergebnisse/m6-t14-verzeichnung-querfarbe.md).

Bewertet Systemdateien mit `mikroptim eval` (Raster, Referenz durch die Austrittspupille, skalare
PSF für den Strehl) zweimal: mit dem idealen Bildpunkt (wie M6-T13) und mit dem Bildpunkt im
Strahlschwerpunkt bei λ_ref (ADR-0019, `--reference-point centroid`; Verzeichnung und Querfarbe
aus den Hauptstrahlen der Bewertung, relativ zum idealen Punkt). Felder wie in der Datei
(0/70/100 % des erreichten Feldes), Kennzahlen über 436–656 nm:

- NA = Pupillenradius / f_soll, Feld = tan θ_max / tan θ_voll (FZ 25), Elemente,
- RMS 0/70/100 %: größter RMS [Wellen bei λ_ref] über die Wellenlängen je Feld,
- Strehl min: kleinster FFT-Strehl über Felder und Wellenlängen,
- Verzeichnung: Hauptstrahl bei λ_ref am größten Feld [%],
- Querfarbe: Spanne der Hauptstrahlhöhen über 436–656 nm am größten Feld [µm] und im
  Verhältnis zum Airy-Radius 0,61 λ_ref/NA.

Aufruf (Wurzelverzeichnis, nach dem Release-Build):
    python tools/m6t14_tabelle.py NAME=SYSTEM.json [NAME=SYSTEM.json ...] --out DIR [--gpu]
                                  [--exe PFAD] [--grid N] [--json DATEI]
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIELD_DEG = math.degrees(math.atan(12.5 / 164.5))  # volles Feld (FZ 25, f_T 164,5 mm)
FOCAL = 3.29
BAND = (435.0, 657.0)  # 436–656 nm


def default_exe():
    for p in ('build/release-cuda/Release/mikroptim.exe', 'build/release-cuda/mikroptim',
              'build/release-cpu/Release/mikroptim.exe', 'build/release-cpu/mikroptim'):
        if os.path.exists(os.path.join(ROOT, p)):
            return os.path.join(ROOT, p)
    return 'mikroptim'


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_air(material):
    return material == 'AIR' or (isinstance(material, dict) and material.get('type') == 'air')


def glass_elements(system):
    """Flächen vor der letzten mit Glas danach (wie glass_elements der Bibliothek)."""
    surf = system['surfaces']
    return sum(1 for i in range(len(surf) - 1) if not is_air(surf[i]['material']))


def run_eval(exe, src, out_json, grid, gpu, extra):
    if os.path.exists(out_json):
        return load(out_json)
    args = [exe, 'eval', src, '--grid', str(grid), '--json', '--psf',
            '--reference', 'exit-pupil'] + extra
    if gpu:
        args.append('--gpu')
    with open(out_json + '.tmp', 'w', encoding='utf-8') as f:
        rc = subprocess.run(args, stdout=f, cwd=ROOT).returncode
    if rc != 0:
        raise RuntimeError(f'eval {src} {extra}: Exit {rc}')
    os.replace(out_json + '.tmp', out_json)
    return load(out_json)


def summarize(ev, na):
    tuples = ev['tuples']
    fields = sorted({t['field'] for t in tuples})
    ref_nm = ev.get('reference_nm') or 546.07
    rms = []
    strehl = []
    for f in fields:
        r = [t['rms_opd_waves'] for t in tuples
             if t['field'] == f and BAND[0] <= t['wavelength_nm'] <= BAND[1]]
        rms.append(max(r) if r else float('nan'))
        for t in tuples:
            if t['field'] == f and BAND[0] <= t['wavelength_nm'] <= BAND[1] and t.get('psf'):
                strehl.append(t['psf']['strehl'])
    last = fields[-1]
    edge = [t for t in tuples if t['field'] == last and BAND[0] <= t['wavelength_nm'] <= BAND[1]]
    chief = [t['chief_ray_mm'][1] for t in edge if t['chief_ray_mm'][1] is not None]
    ref = min(tuples, key=lambda t: abs(t['wavelength_nm'] - ref_nm) + 1e3 * (t['field'] != last))
    lateral_um = (max(chief) - min(chief)) * 1e3 if len(chief) == len(edge) and chief else None
    airy_um = 0.61 * ref_nm * 1e-3 / na if na > 0 else None
    return {'rms': rms, 'strehl_min': min(strehl) if strehl else None,
            'distortion_percent': ref['distortion_chief_percent'],
            'lateral_um': lateral_um,
            'lateral_airy': lateral_um / airy_um if lateral_um is not None and airy_um else None,
            'failed': sum(t['failed_rays'] for t in tuples)}


def fmt(x, d=3):
    return '–' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{d}f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('systems', nargs='+', metavar='NAME=SYSTEM.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--exe', default=default_exe())
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--grid', type=int, default=256)
    ap.add_argument('--json', default=None, help='Zusammenfassung als JSON')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for item in args.systems:
        name, _, path = item.partition('=')
        s = load(path)
        f = s.get('target', {}).get('focal_length', FOCAL)
        na = s['surfaces'][0]['semi_diameter'] / f
        angles = s['source']['field_angles_deg']
        field = math.tan(math.radians(max(abs(a) for a in angles))) / \
            math.tan(math.radians(FIELD_DEG))
        spec = s['spectrum']['wavelengths_nm']
        result = {'name': name, 'file': path, 'na': na, 'field': field,
                  'spectrum': [min(spec), max(spec)], 'elements': glass_elements(s),
                  'reference_point': s.get('reference_point', 'ideal')}
        for key, point in (('ideal', 'ideal'), ('chief', 'centroid')):
            src = os.path.join(args.out, f'{name}_{key}_in.json')
            t = copy.deepcopy(s)
            t['reference_sphere'] = t.get('reference_sphere', 'image_infinite')
            if t['reference_sphere'] == 'last_surface':
                t['reference_sphere'] = 'image_infinite'
            t['reference_point'] = point
            # Kataloge absolut (die Kopie liegt im Ausgabeverzeichnis)
            base = os.path.dirname(os.path.abspath(path))
            cats = []
            for c in t.get('glass_catalogs', []):
                if not os.path.isabs(c):
                    for cand in (os.path.join(base, c), os.path.join(ROOT, c)):
                        if os.path.exists(cand):
                            c = os.path.abspath(cand)
                            break
                cats.append(c)
            t['glass_catalogs'] = cats
            save(t, src)
            ev = run_eval(args.exe, src, os.path.join(args.out, f'{name}_{key}_eval.json'),
                          args.grid, args.gpu, [])
            result[key] = summarize(ev, na)
        rows.append(result)
        print(f'{name}: fertig', file=sys.stderr, flush=True)
    print('| System | NA | Feld | Spektrum | El. | RMS 0/70/100 % (Schwerpunkt) | '
          'RMS 0/70/100 % (ideal) | Strehl min (Schwerpunkt / ideal) | Verzeichnung [%] | '
          'Querfarbe [µm] (× Airy) |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        c, i = r['chief'], r['ideal']
        print(f"| {r['name']} | {fmt(r['na'], 4)} | {fmt(100 * r['field'], 1)} % | "
              f"{r['spectrum'][0]:.0f}–{r['spectrum'][1]:.0f} | {r['elements']} | "
              f"{'/'.join(fmt(x) for x in c['rms'])} | {'/'.join(fmt(x) for x in i['rms'])} | "
              f"{fmt(c['strehl_min'], 2)} / {fmt(i['strehl_min'], 2)} | "
              f"{fmt(c['distortion_percent'], 3)} | {fmt(c['lateral_um'], 3)} "
              f"({fmt(c['lateral_airy'], 2)}) |")
    if args.json:
        save(rows, args.json)


if __name__ == '__main__':
    main()
