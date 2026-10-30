#!/usr/bin/env python3
"""Lauf auf der Zielspezifikation (M6-T6): Stufen nach docs/optimierung.md, Abschnitt 7.

Das Skript verkettet Aufrufe von `mikroptim synth`, `mikroptim optimize`, `mikroptim snap` und
`mikroptim eval` und ändert zwischen den Stufen die Systemdateien (Feld, Spektrum,
Nebenbedingungen, Variablen). Alle Zwischenstände (Systemdateien, Protokolle als JSON und Text)
landen im Ausgabeverzeichnis; eine Stufe, deren Ergebnisdatei schon existiert, wird übersprungen
(Fortsetzung nach Abbruch). Bericht: docs/ergebnisse/m6-t6-zielspezifikation.md.

Aufruf (aus dem Wurzelverzeichnis des Repositorys, nach dem Release-Build):
    python tools/m6t6_zielspezifikation.py [--out DIR] [--exe PFAD] [--gpu] [--bis STUFE]
                                           [--aktuell] [--farbregel] [--farb-budget N] [--farb-stufen N] [--farb-stetig]
                                           [--farbe-frueh]
                                           [--farb-achromat]
                                           [--synth-arg ARG]...

Ohne --aktuell rechnet `synth` wie im dokumentierten Lauf (`--stall 0`, vor M11-T1c). Mit
--aktuell gelten die aktuellen Standards von `synth` (Stagnationsabbruch), mit --farbregel
zusätzlich die Farbregel (M11-T5c, `--color-rule`); --synth-arg reicht weitere Optionen durch.

Die Bausteine (add_gap_variables, set_field, ...) sind auch einzeln nutzbar.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------------------------
# Dateien
# ---------------------------------------------------------------------------------------------

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save(system, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(system, f, indent=2, ensure_ascii=False)
        f.write('\n')


def is_air(material):
    return isinstance(material, str) and material == 'AIR'


def glass_elements(system):
    """Glaselemente wie glass_elements() der Bibliothek: Flächen vor der letzten mit Glas danach."""
    s = system['surfaces']
    return sum(1 for i in range(len(s) - 1) if not is_air(s[i]['material']))


def plate_gaps(system):
    return {g['after_surface'] for g in system.get('ghost_plates', [])}


def thickness_surfaces(system):
    """Dateiflächen, deren Dicke schon Variable ist."""
    out = set()
    S = len(system['surfaces'])
    for v in system.get('variables', {}).get('items', []):
        if v['type'] != 'thickness':
            continue
        if 'surface' in v:
            out.add(v['surface'])
        elif v.get('surfaces') == 'all':
            out.update(range(S - 1))
        else:
            out.update(v.get('surfaces', []))
    return out


# ---------------------------------------------------------------------------------------------
# Änderungen zwischen den Stufen
# ---------------------------------------------------------------------------------------------

def add_gap_variables(system, air=(0.1, 10.0), glass=(0.4, 6.0), skip=()):
    """Luftspalte ohne Geisterplatten (außer dem Blendenspalt) und Glasdicken als Variablen.

    Spalte mit Geisterplatten bleiben fest: Der Einfüge-Operator A teilt genau diese Spalte, und
    ein Teilstück unter der unteren Schranke einer geerbten Dickenvariable wäre ein Formatfehler
    (docs/optimierung.md, Abschnitt 6). Liefert die Liste der neuen Flächen.
    """
    s = system['surfaces']
    have = thickness_surfaces(system)
    gaps = plate_gaps(system)
    new_air, new_glass = [], []
    for i in range(1, len(s) - 1):
        if i in have or i in skip:
            continue
        if is_air(s[i]['material']):
            if i in gaps:
                continue
            new_air.append(i)
        else:
            new_glass.append(i)
    items = system.setdefault('variables', {'items': []})['items']
    pos = next((k for k, v in enumerate(items) if v['type'] == 'ghost_coefficient'), len(items))
    add = []
    if new_air:
        lo = min(air[0], min(s[i]['thickness'] for i in new_air))
        hi = max(air[1], max(s[i]['thickness'] for i in new_air))  # Startwert im Bereich
        add.append({'type': 'thickness', 'surfaces': new_air, 'min': lo, 'max': hi,
                    'comment': 'Luftspalte (M6-T6: zwischen den Stufen freigegeben)'})
    if new_glass:
        lo = min(glass[0], min(s[i]['thickness'] for i in new_glass))
        hi = max(glass[1], max(s[i]['thickness'] for i in new_glass))
        add.append({'type': 'thickness', 'surfaces': new_glass, 'min': lo, 'max': hi,
                    'comment': 'Glasdicken (M6-T6: zwischen den Stufen freigegeben)'})
    items[pos:pos] = add
    return new_air + new_glass


def remove_gap_variables_in_plate_gaps(system):
    """Dickenvariablen von Spalten mit Geisterplatten entfernen (vor einem synth-Lauf)."""
    gaps = plate_gaps(system)
    items = system.get('variables', {}).get('items', [])
    out = []
    for v in items:
        if v['type'] == 'thickness' and 'surfaces' in v and isinstance(v['surfaces'], list):
            keep = [i for i in v['surfaces'] if i not in gaps]
            if not keep:
                continue
            v = dict(v, surfaces=keep)
        elif v['type'] == 'thickness' and v.get('surface') in gaps:
            continue
        out.append(v)
    system['variables']['items'] = out


def reset_ghost_factors(system, point):
    """Gewichte der L1-Strafe auf die Basis zurück (synth schreibt μ und ε des Endpunkts)."""
    mu = point.get('ghost_weight', 1.0)
    eps = point.get('ghost_smoothing', 1.0)
    for c in system.get('constraints', {}).get('items', []):
        if c['type'] == 'ghost_penalty':
            c['weight'] = c.get('weight', 1.0) / mu
            c['smoothing'] = c.get('smoothing', 1e-6) / eps


def set_constraint(system, spec, key=('type',)):
    items = system.setdefault('constraints', {'items': []})['items']
    for k, c in enumerate(items):
        if all(c.get(x) == spec.get(x) for x in key):
            items[k] = spec
            return
    items.append(spec)


def set_field(system, angles, weights):
    system['source']['field_angles_deg'] = list(angles)
    system['source']['field_weights'] = list(weights)


def set_spectrum(system, wavelengths, weights, reference=546.07):
    system['spectrum'] = {'wavelengths_nm': list(wavelengths), 'weights': list(weights),
                          'reference_nm': reference}


def strip_ghost_plates(system):
    """Geisterplatten, ihre Variablen und die L1-Strafe entfernen (Endsystem)."""
    system.pop('ghost_plates', None)
    items = system.get('variables', {}).get('items', [])
    system['variables']['items'] = [v for v in items if v['type'] != 'ghost_coefficient']
    cons = system.get('constraints', {}).get('items', [])
    system['constraints']['items'] = [c for c in cons if c['type'] != 'ghost_penalty']


def remove_dummy_planes(system, working_distance_min=0.3):
    """Hilfsebenen (Planflächen in Luft ohne Geisterplatten) entfernen; ihr Spalt geht an die
    Fläche davor. Ist die letzte Fläche eine Hilfsebene, wird die Fläche davor die Frontfläche
    und ihr Abstand zum Objekt der Arbeitsabstand (Untergrenze working_distance_min)."""
    assert not system.get('ghost_plates'), 'zuerst strip_ghost_plates'
    while True:
        s = system['surfaces']
        j = next((k for k in range(1, len(s)) if s[k]['type'] == 'plane'
                  and is_air(s[k]['material']) and is_air(s[k - 1]['material'])), None)
        if j is None:
            return
        s[j - 1]['thickness'] += s[j]['thickness']
        del s[j]

        def m(i):
            return i if i < j else (j - 1 if i == j else i - 1)

        for v in system.get('variables', {}).get('items', []):
            if 'surface' in v:
                v['surface'] = m(v['surface'])
            if isinstance(v.get('surfaces'), list):
                new = []
                for i in v['surfaces']:
                    if v['type'] == 'thickness' and i in (j, j - 1) and j - 1 >= len(s) - 1:
                        continue  # Spalt wird Arbeitsabstand
                    if m(i) not in new:
                        new.append(m(i))
                v['surfaces'] = new
            if v['type'] == 'working_distance':
                # nicht über den aktuellen Wert (die Synthese erlaubt 0,25 mm)
                v['min'] = min(max(v.get('min', 0.0), working_distance_min),
                               s[-1]['thickness'])
        system['variables']['items'] = [
            v for v in system['variables']['items']
            if not (isinstance(v.get('surfaces'), list) and not v['surfaces'])
            and not (v['type'] == 'thickness' and v.get('surface', -1) >= len(s) - 1)]
        for c in system.get('constraints', {}).get('items', []):
            for key in ('surface', 'reference_surface'):
                if key in c:
                    c[key] = m(c[key])
            if isinstance(c.get('surfaces'), list):
                # Dicken-Bedingungen gelten für den Spalt nach der Fläche (0 … S − 2)
                gap = c['type'] in ('min_center_thickness', 'max_center_thickness',
                                    'min_edge_thickness')
                limit = len(s) - 2 if gap else len(s) - 1
                new = []
                for i in c['surfaces']:
                    if m(i) not in new and m(i) <= limit:
                        new.append(m(i))
                c['surfaces'] = new


def total_length(system, first=0):
    return sum(s['thickness'] for s in system['surfaces'][first:])


# Buchdahl-Fits (3 Terme, 400–700 nm) der frei erfundenen Testgläser im Startsystem; am Ende
# durch den Katalognamen ersetzt (Sellmeier aus data/glass/synthetisch.json)
BUCHDAHL_FITS = {
    'SYN-36': 1.8066761675654839,
    'SYN-30': 1.7923202521223949,
}


def buchdahl_to_catalog(system):
    """Feste Buchdahl-Materialien, die ein Fit eines Testglases sind, durch den Katalognamen
    ersetzen (Glasvariablen bleiben unberührt). Liefert die ersetzten Flächen."""
    glass_vars = set()
    for v in system.get('variables', {}).get('items', []):
        if v['type'] == 'glass':
            if v.get('surfaces') == 'all':
                return []
            glass_vars.update([v['surface']] if 'surface' in v else v.get('surfaces', []))
    out = []
    for i, s in enumerate(system['surfaces']):
        m = s['material']
        if isinstance(m, dict) and m.get('model') == 'buchdahl' and i not in glass_vars:
            for name, n0 in BUCHDAHL_FITS.items():
                if m['n0'] == n0:
                    s['material'] = name
                    out.append((i, name))
    return out


# Zielspezifikation (docs/spezifikation.md) und Annahmen des Laufs (Bericht, Abschnitt Annahmen)
FOCAL = 3.29
FIELD_DEG = math.degrees(math.atan(12.5 / 164.5))        # 4,345° (FZ 25, f_T 164,5 mm)
FIELD_ANGLES = [0.0, math.degrees(math.atan(0.7 * math.tan(math.radians(FIELD_DEG)))), FIELD_DEG]
FIELD_WEIGHTS = [1.0, 1.0, 1.0]
PARFOCAL = 45.0          # Annahme: Anlagefläche in der Blendenebene (offset 0)
PETZVAL_RADIUS = 130.0   # spezifikation.md: R_P >= ca. 130 mm
MAX_SEMI_DIAMETER = 10.0  # Annahme: Linsendurchmesser <= 20 mm
SPECTRA = {
    'mono': ([546.07], [1.0]),
    'zwei': ([480.0, 643.85], [1.0, 1.0]),
    'voll': ([435.83, 486.13, 546.07, 587.56, 656.27], [1.0, 1.0, 1.0, 1.0, 1.0]),
    'erweitert': ([404.66, 435.83, 486.13, 546.07, 587.56, 656.27, 706.52],
                  [0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]),
}


def prepare_field(system, petzval_weight=1e4, parfocal_weight=1.0, diameter_weight=1e4):
    """Feldphase: drei Felder, Petzval-, Längen- und Durchmesserbedingung.

    Länge: Blende bis Objekt höchstens 45 mm (Annahme: die Anlagefläche liegt in der
    Blendenebene oder davor; die Abgleichlänge selbst stellt dann die Fassung ein). Der
    Blendenspalt (Lage der Blende) ist Variable; die Normierungsradien der Geisterplatten decken
    danach das Bündel bis zum Randfeld ab.
    """
    set_field(system, FIELD_ANGLES, FIELD_WEIGHTS)
    s = system['surfaces']
    system.setdefault('target', {})['parfocal_length'] = PARFOCAL
    set_constraint(system, {'type': 'parfocal_length', 'mode': 'max', 'reference_surface': 0,
                            'offset': 0.0, 'weight': parfocal_weight,
                            'comment': 'Annahme: Blende bis Objekt <= 45 mm (Anlagefläche in '
                                       'der Blendenebene oder davor)'})
    set_constraint(system, {'type': 'petzval', 'max_curvature': 1.0 / PETZVAL_RADIUS,
                            'weight': petzval_weight,
                            'comment': 'Petzvalradius >= 130 mm (spezifikation.md)'})
    set_constraint(system, {'type': 'max_semi_diameter', 'semi_diameter': MAX_SEMI_DIAMETER,
                            'clearance': 0.2, 'weight': diameter_weight,
                            'comment': 'Annahme: Linsendurchmesser <= 20 mm (freie Öffnung + 0,2 mm)'})
    items = system['variables']['items']
    if 0 not in thickness_surfaces(system):
        items.insert(0, {'type': 'thickness', 'surface': 0, 'min': 0.5, 'max': 40.0,
                         'comment': 'Blendenspalt (Lage der Blende)'})
    # Normierungsradien: Pupille plus Hauptstrahlhöhe am Ort der Platte
    a = s[0]['semi_diameter']
    tan_f = math.tan(math.radians(FIELD_DEG))
    for g in system.get('ghost_plates', []):
        i = g['after_surface']
        z = sum(x['thickness'] for x in s[:i]) + g['distance']
        if i == 1:      # kollimiert vor der ersten Linse
            g['norm_radius'] = a + z * tan_f
        else:           # konvergentes Bündel vor dem Objekt: Bildhöhe dazu
            g['norm_radius'] = g['norm_radius'] + FOCAL * tan_f


# ---------------------------------------------------------------------------------------------
# Programmaufrufe
# ---------------------------------------------------------------------------------------------

def synth_options(args):
    """Zusatzoptionen der synth-Aufrufe aus --aktuell, --farbregel und --synth-arg."""
    opts = [] if (args.aktuell or args.farbregel or args.farb_achromat) else ['--stall', '0']
    if args.farbregel or args.farb_achromat:
        opts.append('--color-rule')
    if args.farb_achromat:
        opts.append('--split-achromat')
    for a in args.synth_arg:
        opts.extend(a.split())
    return opts


class Runner:
    def __init__(self, exe, out, gpu=False, synth_options=None):
        self.exe = exe
        self.out = out
        self.gpu = gpu
        # Zusatzoptionen für jeden synth-Aufruf; Standard wie im dokumentierten Lauf
        self.synth_options = ['--stall', '0'] if synth_options is None else list(synth_options)
        os.makedirs(out, exist_ok=True)
        # Laufzeiten aller Programmaufrufe, auch über Fortsetzungen hinweg
        self.times_file = os.path.join(out, 'm6t6_zeiten.json')
        self.times = load(self.times_file) if os.path.exists(self.times_file) else {}

    def path(self, name):
        return os.path.join(self.out, name)

    def run(self, name, args, text_name=None):
        text = self.path(text_name or (name + '.txt'))
        t0 = time.time()
        with open(text, 'w', encoding='utf-8') as f:
            r = subprocess.run([self.exe] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        dt = time.time() - t0
        self.times[name] = dt
        save(self.times, self.times_file)
        print(f'{name}: Exit {r.returncode}, {dt:.0f} s', flush=True)
        if r.returncode != 0:
            raise RuntimeError(f'{name}: mikroptim endete mit {r.returncode}, siehe {text}')
        return dt

    def synth(self, name, start, options):
        out = self.path(name + '.json')
        if os.path.exists(out):
            print(f'{name}: vorhanden, übersprungen')
            return out
        args = (['synth', start, '-o', out, '--log', self.path(name + '_log.json')] +
                self.synth_options + options)
        if self.gpu:
            args.append('--gpu')
        self.run(name, args)
        return out

    def optimize(self, name, start, options):
        out = self.path(name + '.json')
        if os.path.exists(out):
            print(f'{name}: vorhanden, übersprungen')
            return out
        args = ['optimize', start, '-o', out] + options
        if self.gpu:
            args.append('--gpu')
        self.run(name, args)
        return out


def endpoint(log_path):
    return load(log_path)['result']['point']


def scale_ghost_weight(system, factor):
    for c in system.get('constraints', {}).get('items', []):
        if c['type'] == 'ghost_penalty':
            c['weight'] = c.get('weight', 1.0) * factor


def restore_default_selections(before, after):
    """`mikroptim insert` schreibt die Standardauswahl mechanischer Bedingungen als explizite
    Liste der alten Flächen (renumber_surfaces), damit sich die Merit nicht ändert; neue Flächen
    wären dann nicht eingeschränkt. Wie finish_insertion des Controllers: Einträge, die vorher
    die Standardauswahl hatten, bekommen sie zurück."""
    cb = before.get('constraints', {}).get('items', [])
    ca = after.get('constraints', {}).get('items', [])
    for b, a in zip(cb, ca):
        if b['type'] == a['type'] and 'surfaces' not in b and 'surface' not in b:
            a.pop('surfaces', None)


def merit_after(text_path):
    """Merit nach der Optimierung aus der Textausgabe von `mikroptim optimize`."""
    with open(text_path, encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('Merit nachher [Wellen'):
                return float(line.split()[-1])
    return math.inf


def forced_insertion(runner, tag, src, relax_options, glass='SYN-36', thickness=0.4,
                     min_gap=0.7, curvature_bound=1.2, log=print):
    """Erzwungenes Einfügen, wenn der Controller ohne aktive Geisterplatte stehen bleibt.

    Für jeden Luftspalt ohne Geisterplatten (außer dem Blendenspalt) mit Dicke >= min_gap wird
    mit `mikroptim insert` eine Planplatte (c₀ = 0) der Dicke `thickness` aus `glass` in die
    Mitte gesetzt (Krümmungen variabel in ±curvature_bound, Dicke variabel), danach alle
    Kandidaten nachoptimiert; der mit der kleinsten Merit gewinnt. Liefert (Datei, Liste der
    Kandidaten) oder (None, []) ohne passenden Spalt.
    """
    s = load(src)
    gaps = plate_gaps(s)
    cands = []
    for i in range(1, len(s['surfaces']) - 1):
        t = s['surfaces'][i]['thickness']
        if is_air(s['surfaces'][i]['material']) and i not in gaps and t >= min_gap:
            cands.append(i)
    results = []
    for i in cands:
        t = s['surfaces'][i]['thickness']
        name = f'{tag}_ins{i}'
        ins = runner.path(name + '_raw.json')
        if not os.path.exists(ins):
            try:
                runner.run(name + '_insert',
                           ['insert', src, '--meniscus',
                            f'{i},{(t - thickness) / 2!r},0,{glass}', '--thickness',
                            repr(thickness), '-o', ins])
            except RuntimeError as e:
                # Spalt nicht nutzbar (z. B. Flächentabelle voll): nächsten Kandidaten versuchen
                # statt den ganzen Lauf abzubrechen (v9: p3_r9_ins17_insert)
                log(f'  Spalt {i} übersprungen: {e}')
                continue
        x = load(ins)
        restore_default_selections(s, x)
        items = x['variables']['items']
        for v in items:
            if v['type'] == 'curvature' and v.get('surfaces') == [i + 1, i + 2]:
                v['min'], v['max'] = -curvature_bound, curvature_bound
        if i + 1 not in thickness_surfaces(x):
            items.append({'type': 'thickness', 'surface': i + 1, 'min': thickness, 'max': 5.0,
                          'comment': 'Mittendicke (M6-T6: erzwungenes Einfügen)'})
        add_gap_variables(x)
        prep = runner.path(name + '_in.json')
        save(x, prep)
        try:
            out = runner.optimize(name, prep, relax_options)
        except RuntimeError as e:
            log(f'  Spalt {i} übersprungen: {e}')
            continue
        f = merit_after(runner.path(name + '.txt'))
        results.append({'gap': i, 'file': out, 'merit': f})
        log(f'  erzwungen: Spalt {i}, F = {f:.4e}')
    if not results:
        return None, []
    best = min(results, key=lambda r: r['merit'])
    return best['file'], results


def forced_split(runner, tag, src, relax_options, min_thickness=1.0, log=print):
    """Erzwungenes Kittglied (Operator B), wenn der Controller in einer Spektrumphase keinen
    Operator B empfiehlt: jede Glaslinse mit Mittendicke >= min_thickness wird mit
    `mikroptim insert --split` in der Mitte geteilt (Kittfläche = Mittel der Außenkrümmungen),
    das Glas des hinteren Teils wird Variable (Glas-Hülle mit Standardgewicht, falls sie fehlt);
    alle Kandidaten werden nachoptimiert, der mit der kleinsten Merit gewinnt."""
    s = load(src)
    surf = s['surfaces']
    cands = [i for i in range(1, len(surf) - 1)
             if not is_air(surf[i]['material']) and surf[i]['thickness'] >= min_thickness]
    results = []
    for i in cands:
        cs = 0.5 * (surf[i].get('curvature', 0.0) + surf[i + 1].get('curvature', 0.0))
        name = f'{tag}_split{i}'
        raw = runner.path(name + '_raw.json')
        if not os.path.exists(raw):
            try:
                runner.run(name + '_insert', ['insert', src, '--split', f'{i},{cs!r},0.5',
                                              '--free-glass', '-o', raw])
            except RuntimeError as e:
                log(f'  Linse {i} übersprungen: {e}')
                continue
        x = load(raw)
        restore_default_selections(s, x)
        items = x['variables']['items']
        for v in items:
            if v['type'] == 'curvature' and v.get('surfaces') == [i + 1] and 'min' not in v:
                v['min'], v['max'] = -1.2, 1.2
        if not any(c['type'] == 'glass_hull' for c in x['constraints']['items']):
            x['constraints']['items'].append({'type': 'glass_hull',
                                              'comment': 'M6-T6: erzwungenes Kittglied'})
        add_gap_variables(x)
        prep = runner.path(name + '_in.json')
        save(x, prep)
        try:
            out = runner.optimize(name, prep, relax_options)
        except RuntimeError as e:
            log(f'  Linse {i} übersprungen: {e}')
            continue
        f = merit_after(runner.path(name + '.txt'))
        results.append({'lens': i, 'file': out, 'merit': f})
        log(f'  erzwungen (B): Linse {i}, F = {f:.4e}')
    if not results:
        return None, []
    best = min(results, key=lambda r: r['merit'])
    return best['file'], results


def synth_loop(runner, tag, start, synth_options, relax_options, progress, done_value,
               max_rounds=8, min_mu_scale=0.5, log=print, restart=None, forced=0,
               forced_kind='meniscus', element_budget=None, stage_budget=None):
    """synth → (Spalte freigeben und optimieren → synth)* bis zum Ziel.

    Der Homotopie-Controller hält Luftspalte ohne Dickenvariable fest; neue Spalte hinter
    eingefügten Elementen haben keine. Bleibt er stehen (Schrittweite unter Δ_min ohne aktive
    Geisterplatte, Ziel verfehlt), werden die Spalte ohne Geisterplatten Variablen
    (add_gap_variables), das System am erreichten Punkt nachoptimiert und synth ab dort neu
    gestartet. Kommt der Parameter (progress(point)) dabei nicht voran, wird μ aller
    Geisterplatten halbiert (Bedarf früher sichtbar), höchstens bis min_mu_scale.
    synth_options ist eine Liste oder eine Funktion point -> Liste (point = None in der ersten
    Runde, sonst der Endpunkt der vorigen); restart(system, point) passt die Datei vor dem
    Neustart an (z. B. Feldwinkel zurück auf 100 %, weil synth die des Endpunkts schreibt).
    Liefert (Ergebnisdatei, Liste der Runden).
    """
    def finish(out, point):
        # Ergebnis für die nächste Stufe: Gewichte der L1-Strafe auf der Basis, Datei auf der
        # Basis des Neustarts (z. B. Feldwinkel 100 %)
        s = load(out)
        reset_ghost_factors(s, point)
        if restart:
            restart(s, point)
        path = runner.path(f'{tag}_final.json')
        save(s, path)
        return path

    cur = start
    rounds = []
    mu_scale = 1.0
    # stage_budget: höchstens so viele Elemente mehr als zu Beginn der Stufe, über alle Runden
    # (element_budget gilt pro synth-Lauf; in den Farbstufen wuchs das System so auf 30 Elemente)
    stage_limit = glass_elements(load(start)) + stage_budget if stage_budget else None
    last = None
    prev_point = None
    for r in range(max_rounds):
        name = f'{tag}_r{r}'
        opts = synth_options(prev_point) if callable(synth_options) else synth_options
        if element_budget:
            # höchstens element_budget Einfügungen pro synth-Lauf (die letzte Angabe gilt)
            opts = list(opts) + ['--max-elements', str(glass_elements(load(cur)) + element_budget)]
        if stage_limit is not None:
            opts = list(opts) + ['--max-elements', str(stage_limit)]
        out = runner.synth(name, cur, opts)
        res = load(runner.path(name + '_log.json'))['result']
        point = res['point']
        info = {'round': r, 'synth': name, 'status': res['status'], 'point': point,
                'elements': res['elements'], 'insertions': res['insertions'],
                'goal': res.get('goal'), 'mu_scale': mu_scale,
                'seconds': runner.times.get(name)}
        rounds.append(info)
        log(f"  {name}: {res['status']}, Punkt {progress(point):.4f}, {res['elements']} Elemente")
        if res['status'] == 'goal_reached':
            return finish(out, point), rounds
        s = load(out)
        reset_ghost_factors(s, point)
        prev_point = point
        p = progress(point)
        force = False
        if last is not None and p <= last + 1e-3:
            mu_scale *= 0.5
            if mu_scale < min_mu_scale:
                if forced <= 0:
                    log('  kein Fortschritt, Abbruch')
                    return finish(out, point), rounds
                if stage_limit is not None and glass_elements(load(out)) >= stage_limit:
                    log('  Elementbudget der Stufe erschöpft, Abbruch')
                    return finish(out, point), rounds
                force = True
                forced -= 1
                scale_ghost_weight(s, 1.0 / (2.0 * mu_scale))  # μ zurück auf die Basis
                mu_scale = 1.0
                log('  kein Fortschritt: erzwungenes Einfügen')
            else:
                scale_ghost_weight(s, 0.5)
                log(f'  kein Fortschritt: mu x{mu_scale}')
        last = max(p, last) if last is not None else p
        info['relaxed_gaps'] = add_gap_variables(s)
        prep = runner.path(f'{tag}_r{r}_relax_in.json')
        save(s, prep)
        cur = runner.optimize(f'{tag}_r{r}_relax', prep, relax_options)
        info['relax'] = f'{tag}_r{r}_relax'
        if force:
            fn = forced_split if forced_kind == 'split' else forced_insertion
            best, cands = fn(runner, f'{tag}_r{r}', cur, relax_options, log=log)
            info['forced'] = cands
            if best is None:
                log('  kein Spalt für ein erzwungenes Einfügen, Abbruch')
                return finish(out, point), rounds
            cur = best
            last = None
        if restart:
            # die Nachoptimierung lief am Endpunkt (Datei von synth); für den Neustart die
            # Datei wieder auf die Basis bringen (z. B. Feldwinkel 100 %)
            s = load(cur)
            restart(s, point)
            cur = runner.path(f'{tag}_r{r}_restart.json')
            save(s, cur)
    return cur, rounds


def last_point(runner, rounds):
    return load(runner.path(rounds[-1]['synth'] + '_log.json'))['result']['point']


def has_glass_variables(system):
    return any(v['type'] == 'glass' for v in system.get('variables', {}).get('items', []))


def polish(runner, name, src, relax_options, spectrum='erweitert'):
    """Feinoptimierung des realen Systems: Geisterplatten und Hilfsebenen entfernen, Testgläser
    mit Katalognamen, Spektrum mit 405/700 nm schwach, alle Spalte und Glasdicken frei; die
    Glas-Hülle (falls Glasvariablen) in drei Stufen ×10 bis zum Gewicht der Datei."""
    s = load(src)
    strip_ghost_plates(s)
    remove_dummy_planes(s)
    buchdahl_to_catalog(s)
    wl, w = SPECTRA[spectrum]
    set_spectrum(s, wl, w)
    add_gap_variables(s)
    hull = [c for c in s['constraints']['items'] if c['type'] == 'glass_hull']
    stages = [1e-2, 1e-1, 1.0] if hull and has_glass_variables(s) else [1.0]
    w0 = [c.get('weight', 1e4) for c in hull]
    cur = None
    for k, f in enumerate(stages):
        for c, w in zip(hull, w0):
            c['weight'] = w * f
        prep = runner.path(f'{name}_s{k}_in.json')
        save(s, prep)
        cur = runner.optimize(f'{name}_s{k}', prep, relax_options)
        s = load(cur)
        hull = [c for c in s['constraints']['items'] if c['type'] == 'glass_hull']
    return cur


def evaluate(runner, name, src, grid=256):
    """mikroptim eval: Raster mit Referenz durch die Austrittspupille und durch den Scheitel,
    skalare und vektorielle PSF (JSON)."""
    sub = 'e' + 'val'
    variants = {
        'pupille': [],
        'scheitel': ['--reference', 'vertex'],
        'psf': ['--psf'],
        'vektor': ['--vector', '--polarization', 'unpolarized'],
    }
    out = {}
    for key, extra in variants.items():
        js = runner.path(f'{name}_{key}.json')
        if not os.path.exists(js):
            args = [runner.exe, sub, src, '--grid', str(grid), '--json'] + extra
            if runner.gpu:
                args.append('--gpu')
            t0 = time.time()
            with open(js, 'w', encoding='utf-8') as f, \
                    open(runner.path(f'{name}_{key}_stderr.txt'), 'w', encoding='utf-8') as e:
                rc = subprocess.run(args, stdout=f, stderr=e, cwd=ROOT).returncode
            runner.times[f'{name}_{key}'] = time.time() - t0
            save(runner.times, runner.times_file)
            print(f'{name}_{key}: Exit {rc}', flush=True)
            if rc != 0:
                raise RuntimeError(f'{name}_{key}: mikroptim endete mit {rc}')
        out[key] = js
    return out


def fmt(x, nd=4):
    if x is None:
        return '–'
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return '–'
    return f'{x:.{nd}f}'.replace('.', ',')


def surface_table(system, heights=None):
    """Markdown-Tabelle der Flächen (Radius, Dicke, Material, freie Öffnung)."""
    rows = ['| Fläche | Typ | Radius [mm] | Dicke [mm] | Material danach | Strahlhöhe + 0,2 mm |',
            '|---|---|---|---|---|---|']
    for i, s in enumerate(system['surfaces']):
        c = s.get('curvature', 0.0)
        rad = '∞' if c == 0.0 else fmt(1.0 / c, 3)
        m = s['material']
        if isinstance(m, dict):
            m = (f"Buchdahl n₀ = {m['n0']:.5f}, ν = {m['nu'][0]:.5f}/{m['nu'][1]:.5f}"
                 .replace('.', ','))
        h = '–' if not heights or i not in heights else fmt(heights[i], 3)
        rows.append(f"| {i} | {s['type']} | {rad} | {fmt(s['thickness'], 4)} | {m} | {h} |")
    return '\n'.join(rows)


def mechanical_heights(optimize_text):
    """Ist-Werte von max_semi_diameter (größte Strahlhöhe + Zugabe) aus der Textausgabe von
    mikroptim optimize (nachher)."""
    out = {}
    with open(optimize_text, encoding='utf-8') as f:
        for line in f:
            p = line.split()
            if len(p) >= 7 and p[1] == 'max_semi_diameter':
                try:
                    out[int(p[2])] = float(p[6])
                except ValueError:
                    pass
    return out


def tuple_table(eval_json, psf_json=None, vec_json=None):
    """RMS (Wellen bei der Wellenlänge des Tupels), Strehl nach Maréchal, FFT und vektoriell."""
    e = load(eval_json)
    psf = load(psf_json)['tuples'] if psf_json else None
    vec = load(vec_json)['tuples'] if vec_json else None
    rows = ['| Feld [°] | λ [nm] | RMS [Wellen bei λ] | RMS ohne Kippung | Strehl (Maréchal) '
            '| Strehl (FFT, skalar) | Strehl (vektoriell, unpolarisiert) | Verzeichnung [%] | versagt |',
            '|---|---|---|---|---|---|---|---|---|']
    worst = 0.0
    for k, t in enumerate(e['tuples']):
        lam = t['wavelength_nm'] * 1e-6
        rms = t['rms_opd_mm'] / lam if t['rms_opd_mm'] is not None else None
        tilt = (t['rms_opd_tilt_waves'] * e['reference_nm'] / t['wavelength_nm']
                if t['rms_opd_tilt_waves'] is not None else None)
        if rms is not None:
            worst = max(worst, rms)
        s_fft = psf[k]['psf']['strehl'] if psf and psf[k].get('psf') else None
        s_vec = vec[k]['psf']['strehl'] if vec and vec[k].get('psf') else None
        rows.append(f"| {fmt(t['field_angle_deg'], 3)} | {fmt(t['wavelength_nm'], 2)} | "
                    f"{fmt(rms)} | {fmt(tilt)} | {fmt(t['strehl_marechal'], 3)} | "
                    f"{fmt(s_fft, 3)} | {fmt(s_vec, 3)} | "
                    f"{fmt(t['distortion_chief_percent'], 3)} | {t['failed_rays']} |")
    return '\n'.join(rows), worst


def tables(out_dir):
    """Tabellen für den Bericht aus einem fertigen Lauf (Markdown auf stdout)."""
    run = load(os.path.join(out_dir, 'm6t6_lauf.json'))
    print('## Stufen\n')
    print('| Stufe | Runden | Status der letzten Runde | Punkt (NA, Feld) | Glaselemente | '
          'Einfügungen (synth) | erzwungen | Laufzeit [s] |')
    print('|---|---|---|---|---|---|---|---|')
    tf = os.path.join(out_dir, 'm6t6_zeiten.json')
    times = load(tf) if os.path.exists(tf) else run.get('times', {})
    for key, st in run['stages'].items():
        rounds = st.get('rounds')
        if not rounds:
            continue
        last = rounds[-1]
        ins = sum(r['insertions'] for r in rounds)
        forced = sum(1 for r in rounds if r.get('forced'))
        t = sum(v for k, v in times.items() if k.startswith(key + '_'))
        print(f"| {key} | {len(rounds)} | {last['status']} | {fmt(last['point']['aperture'])}, "
              f"{fmt(last['point']['field'], 3)} | {last['elements']} | {ins} | {forced} | "
              f"{t:.0f} |")
    print(f"\nGesamt: {sum(times.values()):.0f} s\n")
    # Fokus in der Bildebene? (Falle F12: Fokus im Scheitel der Hilfsebene vor dem Objekt)
    exe = os.path.join(ROOT, 'build', 'release-cuda', 'Release', 'mikroptim.exe')
    print('| Stufe | Ergebnisdatei | hintere Schnittweite [mm] | Abstand letzte Fläche – Objekt [mm] '
          '| Petzvalradius [mm] |')
    print('|---|---|---|---|---|')
    for key, st in run['stages'].items():
        path = st.get('result') or st.get('final')
        if not path or not os.path.exists(path):
            continue
        out = subprocess.run([exe, 'trace', path, '--json'], capture_output=True, text=True,
                             encoding='utf-8').stdout
        par = json.loads(out)['paraxial']
        w = load(path)['surfaces'][-1]['thickness']
        print(f"| {key} | {os.path.basename(path)} | {fmt(par['back_focal_distance'])} | "
              f"{fmt(w)} | {fmt(par['petzval_radius'], 1)} |")
    print()
    p8 = run['stages'].get('p8')
    if p8:
        for key in ('eval', 'eval_full_field'):
            if key in p8:
                ev = p8[key]
                txt, worst = tuple_table(ev['pupille'], ev['psf'], ev['vektor'])
                print(f'## {key} (Referenz Austrittspupille), größter RMS {fmt(worst)}\n')
                print(txt)
                txt, worst = tuple_table(ev['scheitel'])
                print(f'\n## {key} (Referenz Scheitel), größter RMS {fmt(worst)}\n')
                print(txt)
                print()
        final = load(p8['final'])
        print('## Flächen des Endsystems\n')
        print(surface_table(final))


def aperture_branch(args):
    """Vergleichszweig: Apertur zuerst (Standardfolge von optimierung.md §7) bei 10 % Feld."""
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.out), gpu=args.gpu,
               synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    s = load(args.start)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    start = r.path('a0_start.json')
    save(s, start)

    def full_field(system, point):
        set_field(system, FIELD_ANGLES, FIELD_WEIGHTS)

    t0 = time.time()
    a1, ra = synth_loop(r, 'a1', start,
                        ['--target-na', '0.95', '--start-field', '0.1', '--target-field', '0.1',
                         '--curvature-bound', '1.2', '--max-steps', '3000', '--max-iterations',
                         '500', '--max-elements', '14', '--beam', '2', '--alternatives', '2']
                        + rings, relax, lambda p: p['aperture'], 0.95, restart=full_field,
                        max_rounds=12)
    pt = last_point(r, ra)
    s = load(a1)
    strip_ghost_plates(s)
    remove_dummy_planes(s)
    buchdahl_to_catalog(s)
    set_field(s, [a * pt['field'] for a in FIELD_ANGLES], FIELD_WEIGHTS)
    add_gap_variables(s)
    prep = r.path('a2_in.json')
    save(s, prep)
    a2 = r.optimize('a2', prep, ['--max-iterations', '2000', '-q'] + rings)
    s = load(a2)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    full = r.path('a3_vollfeld_in.json')
    save(s, full)
    summary = {'start': args.start, 'rounds': ra, 'point': pt, 'final': a2,
               'eval': evaluate(r, 'a3', a2), 'eval_full_field': evaluate(r, 'a3_vollfeld', full),
               'times': r.times, 'seconds_total': time.time() - t0}
    with open(r.path('m6t6_apertur_zweig.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print('fertig:', a2)
    return 0


def archive(out_dir):
    """Ausgabeverzeichnis für das Repository verkleinern: behalten werden die Zusammenfassungen,
    die Protokolle der synth-Läufe (Text und JSON), die Ergebnisdateien jeder Stufe, die Texte
    der Feinoptimierung und des Einrastens sowie die Bewertungen; absolute Pfade werden relativ
    zum Wurzelverzeichnis geschrieben. Die übrigen Zwischenstände (Nachoptimierungen, Kandidaten
    der erzwungenen Einfügungen) entstehen beim Wiederholen des Laufs neu."""
    import re
    keep = re.compile(r'^(m6t6_.*\.json|[pa]\d+_r\d+_log\.json|[pa]\d+_r\d+\.txt|'
                      r'p0_start\.json|a0_start\.json|p6_s\d\.txt|p6_s2\.json|'
                      r'p7\.(json|txt)|z2_s0\.(json|txt)|a2\.(json|txt)|'
                      r'(p8|z2|a3)(_vollfeld)?_(pupille|scheitel|psf|vektor)\.json)$')
    # Ergebnisdateien der Stufen (auch Neustart-Dateien, wenn eine Stufe dort endete)
    referenced = set()
    for summary in ('m6t6_lauf.json', 'm6t6_apertur_zweig.json'):
        p = os.path.join(out_dir, summary)
        if os.path.exists(p):
            for m in re.finditer(r'"(?:result|final)": "([^"]+)"', open(p, encoding='utf-8').read()):
                referenced.add(os.path.basename(m.group(1).replace('\\\\', '/').replace('\\', '/')))
    removed = 0
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if name not in referenced and (not keep.match(name) or re.match(r'.*_r\d+_relax', name)):
            os.remove(path)
            removed += 1
            continue
        with open(path, encoding='utf-8') as f:
            text = f.read()
        root = ROOT
        for variant in (root + os.sep, root.replace('\\', '\\\\') + '\\\\', root + '/',
                        root.replace('\\', '/') + '/'):
            text = text.replace(variant, '')
        text = text.replace('\\\\', '/') if name.endswith('.json') else text
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
    print(f'{removed} Dateien entfernt, {len(os.listdir(out_dir))} behalten')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--archiv':
        archive(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'docs', 'ergebnisse',
                                                                    'm6-t6'))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == '--tabellen':
        tables(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'docs', 'ergebnisse',
                                                                   'm6-t6'))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'ergebnisse', 'm6-t6'))
    default_exe = os.path.join(ROOT, 'build', 'release-cuda', 'mikroptim')
    if not os.path.exists(default_exe):  # Visual Studio (Windows)
        default_exe = os.path.join(ROOT, 'build', 'release-cuda', 'Release', 'mikroptim.exe')
    ap.add_argument('--exe', default=default_exe)
    ap.add_argument('--aktuell', action='store_true',
                    help='aktuelle Standards von synth (Stagnationsabbruch) statt --stall 0')
    ap.add_argument('--farbregel', action='store_true',
                    help='Farbregel (synth --color-rule, M11-T5c); setzt --aktuell')
    ap.add_argument('--farb-achromat', action='store_true',
                    help='Operator B mit Achromat-Start (synth --split-achromat); setzt --farbregel')
    ap.add_argument('--farbe-frueh', action='store_true',
                    help='Farbstufen vor der Aperturstufe 3 (bei NA 0,5), dann polychromatisch weiter')
    ap.add_argument('--farb-stetig', action='store_true',
                    help='stetige Spektrum-Homotopie in den Farbstufen (synth --spectrum-continuous)')
    ap.add_argument('--farb-stufen', type=int, default=0,
                    help='Spektrum-Homotopie in den Farbstufen: synth --spectrum-stages N (0: aus)')
    ap.add_argument('--farb-budget', type=int, default=1,
                    help='zusätzliche Elemente je Farbstufe über alle Runden (Standard 1)')
    ap.add_argument('--synth-arg', action='append', default=[],
                    help='weitere Option für jeden synth-Aufruf (mehrfach möglich)')
    ap.add_argument('--start', default=os.path.join(ROOT, 'data', 'systems',
                                                    'zielspezifikation_start.json'))
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--na-feld', type=float, default=0.5,
                    help='NA, bei der das Feld von 10 auf 100 %% geöffnet wird')
    ap.add_argument('--apertur-zweig', action='store_true',
                    help='nur der Vergleichszweig „Apertur zuerst“: NA 0,3 -> 0,95 bei 10 %% Feld '
                         'mit Strahlsuche (B = 2, 2 Varianten), Bewertung bei 10 und 100 %% Feld')
    args = ap.parse_args()
    if args.apertur_zweig:
        return aperture_branch(args)
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.out), gpu=args.gpu,
               synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    common = ['--curvature-bound', '1.2', '--max-steps', '3000', '--max-iterations', '500',
              '--max-elements', '24'] + rings
    summary = {'start': args.start, 'stages': {}}

    def full_field(system, point):
        set_field(system, FIELD_ANGLES, FIELD_WEIGHTS)

    def stage(key, value):
        summary['stages'][key] = value
        summary['times'] = r.times
        with open(r.path('m6t6_lauf.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=1, ensure_ascii=False)

    t0 = time.time()
    # Stufe 1: monochromatisch, Apertur 0,3 -> na_feld bei 10 % Feld (Petzval aktiv)
    s = load(args.start)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    start = r.path('p0_start.json')
    save(s, start)
    na1 = repr(args.na_feld)
    p1, r1 = synth_loop(r, 'p1', start,
                        ['--target-na', na1, '--start-field', '0.1', '--target-field', '0.1']
                        + common, relax, lambda p: p['aperture'], args.na_feld,
                        restart=full_field, max_rounds=12, forced=2)
    stage('p1', {'result': p1, 'rounds': r1})
    pt = last_point(r, r1)

    # Stufe 2: Feld 10 % -> 100 % bei dieser NA
    def opts2(point):
        f = 0.1 if point is None else point['field']
        return ['--target-na', repr(pt['aperture']), '--start-field', repr(f),
                '--target-field', '1'] + common

    p2, r2 = synth_loop(r, 'p2', p1, opts2, relax, lambda p: p['field'], 1.0,
                        restart=full_field, max_rounds=12, forced=2)
    stage('p2', {'result': p2, 'rounds': r2})
    pt = last_point(r, r2)
    field = pt['field']

    # Stufe 3: Apertur -> 0,95 beim erreichten Feld
    def color_stages(cur, na):
        # Stufen 4 und 5: zwei Wellenlängen, dann 436–656 nm am Punkt (na, field)
        for key, spec in [('p4', 'zwei'), ('p5', 'voll')]:
            s = load(cur)
            wl, w = SPECTRA[spec]
            set_spectrum(s, wl, w)
            inp = r.path(f'{key}_in.json')
            save(s, inp)
            stufen = (['--spectrum-continuous'] if args.farb_stetig else
                      ['--spectrum-stages', str(args.farb_stufen)] if args.farb_stufen > 0 else [])
            cur, rr = synth_loop(r, key, inp,
                                 ['--target-na', repr(na), '--start-field', repr(field),
                                  '--target-field', repr(field)] + stufen + common, relax,
                                 lambda p: 0.0, 1.0, restart=full_field, max_rounds=4, forced=1,
                                 forced_kind='split', stage_budget=args.farb_budget)
            stage(key, {'result': cur, 'rounds': rr, 'spectrum': spec, 'aperture': na})
        return cur

    p3_in = p2
    if args.farbe_frueh:
        # Farbe vor der Aperturstufe (bei NA und Feld des Endes von Stufe 2): Bei NA 0,605 hat
        # das monochromatische System keinen Abstand zur Schwelle mehr (Lauf v8: F_B 0,0051 schon
        # bei Spektrumbreite 1/1024), bei NA 0,5 schon
        p3_in = color_stages(p2, last_point(r, r2)['aperture'])
    p3, r3 = synth_loop(r, 'p3', p3_in,
                        ['--target-na', '0.95', '--start-field', repr(field), '--target-field',
                         repr(field)] + common, relax, lambda p: p['aperture'], 0.95,
                        restart=full_field, max_rounds=16, forced=3, element_budget=3)
    stage('p3', {'result': p3, 'rounds': r3})
    pt = last_point(r, r3)
    na = pt['aperture']
    cur = p3 if args.farbe_frueh else color_stages(p3, na)

    # Stufe 6: Feinoptimierung ohne Geisterplatten, 405/700 nm schwach dazu, am erreichten Punkt
    s = load(cur)
    set_field(s, [a * field for a in FIELD_ANGLES], FIELD_WEIGHTS)
    p6_in = r.path('p6_punkt.json')
    save(s, p6_in)
    p6 = polish(r, 'p6', p6_in, ['--max-iterations', '2000', '-q'] + rings)
    stage('p6', {'result': p6})
    final = p6
    # Stufe 7: Glas-Einrasten (nur mit Glasvariablen)
    if has_glass_variables(load(p6)):
        out = r.path('p7.json')
        if not os.path.exists(out):
            a = ['snap', p6, '-o', out, '--neighbors', '3', '--max-iterations', '500'] + rings
            if r.gpu:
                a.append('--gpu')
            r.run('p7', a)
        final = out
        stage('p7', {'result': out})
    # Stufe 8: Bewertung am erreichten Punkt (NA na, Feldanteil field); bei field < 1 auch mit
    # den Feldwinkeln der Spezifikation
    info = {'final': final, 'aperture': na, 'field': field,
            'eval': evaluate(r, 'p8', final)}
    if field < 1.0:
        s = load(final)
        set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
        full = r.path('p8_vollfeld_in.json')
        save(s, full)
        info['eval_full_field'] = evaluate(r, 'p8_vollfeld', full)
    info['seconds_total'] = time.time() - t0
    stage('p8', info)

    # Zwischenkandidat: Ende der Feldstufe (volles Feld, monochromatisch), ohne Geisterplatten
    # nachoptimiert und bewertet
    s = load(p2)
    z2_in = r.path('z2_punkt.json')
    save(s, z2_in)
    z2 = polish(r, 'z2', z2_in, ['--max-iterations', '2000', '-q'] + rings, spectrum='mono')
    stage('z2', {'result': z2, 'eval': evaluate(r, 'z2', z2)})
    print('fertig:', final)
    return 0


if __name__ == '__main__':
    sys.exit(main())
