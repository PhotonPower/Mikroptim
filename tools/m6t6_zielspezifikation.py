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
                                           [--farbe-frueh] [--farbe-von-anfang zwei|voll]
                                           [--farb-achromat]
                                           [--referenzsphaere K] [--sinusbedingung W]
                                           [--osc-versuch [--osc-gewicht W]
                                            [--ohne-f12-schutz | --mit-f12-schutz]]
                                           [--synth-arg ARG]...

Ohne --aktuell rechnet `synth` wie im dokumentierten Lauf (`--stall 0`, vor M11-T1c). Mit
--aktuell gelten die aktuellen Standards von `synth` (Stagnationsabbruch), mit --farbregel
zusätzlich die Farbregel (M11-T5c, `--color-rule`); --synth-arg reicht weitere Optionen durch.

Referenzsphäre (M6-T6b): Standard ist `image_infinite` (ADR-0017), die Startdatei bekommt das Feld
`reference_sphere`. Die Läufe vor M6-T7 (Bericht M6-T6, M11-T4) rechnet
`--referenzsphaere last_surface` nach; dann gilt im OSC-Versuch auch wieder der F12-Schutz.
Bericht des Laufs mit neuer Konvention: docs/ergebnisse/m6-t6b-zielspezifikation-adr0017.md.

Zyklischer Ablauf (M6-T10, Bericht docs/ergebnisse/m6-t10-zyklisch.md): mit --zyklus
feld,farbe,apertur und einem realen System als --start (z. B. Z2 nach dem Rückbau aus M6-T9)
je Stufe Rückkehr zu Geisterplatten (`mikroptim insert --ghost-return`), Stufe, Z2, Rückbau am
realen System und Z2 danach; Tabelle mit `--zyklus-tabelle DIR`.

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
        # vereinigte Spalte über Einträge hinweg nur einmal (M6-T9: nach dem Rückbau können
        # beide Spalte um eine Hilfsebene Dickenvariablen haben, sonst „Variable doppelt“)
        seen = {}
        for v in system.get('variables', {}).get('items', []):
            if v['type'] not in ('thickness', 'curvature', 'glass'):
                continue
            done = seen.setdefault(v['type'], set())
            if isinstance(v.get('surfaces'), list):
                v['surfaces'] = [i for i in v['surfaces'] if i not in done]
                done.update(v['surfaces'])
            elif 'surface' in v:
                if v['surface'] in done:
                    v['surfaces'] = []  # Eintrag entfällt unten
                    del v['surface']
                else:
                    done.add(v['surface'])
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
    # M6-T11: Operator B mit Rang nach dem spektralen Rest und Startradien des Glieds, Farbbedarf
    # aus den chromatischen Seidel-Termen, Kittglied-Modul statt Meniskus
    if getattr(args, 'farb_spektrum_rang', False):
        opts.append('--split-spectrum')
    if getattr(args, 'farb_startradien', False):
        opts.append('--split-member-start')
    if getattr(args, 'farb_chromatisch', None):
        opts += ['--color-chromatic', repr(args.farb_chromatisch)]
    if getattr(args, 'farb_modul', None):
        opts += ['--color-module', args.farb_modul]
        if getattr(args, 'farb_modul_immer', False):
            opts.append('--color-module-always')
        if getattr(args, 'farb_anomal', False):
            opts.append('--color-module-anomalous')
    if getattr(args, 'sinusbedingung', None):
        opts += ['--sine-condition', repr(args.sinusbedingung)]
    if getattr(args, 'feldregel', False):
        opts.append('--field-rule')  # M6-T8
    if getattr(args, 'einfuegen_verbessern', None):
        opts += ['--insertion-improve', str(args.einfuegen_verbessern)]  # M6-T8
    # M6-T9: Rückbau (nach den Phasen) und Einstellungen (auch für das Elementbudget)
    if getattr(args, 'rueckbau', False) or getattr(args, 'nur_rueckbau', False):
        opts.append('--prune')
    if getattr(args, 'rueckbau', False) or getattr(args, 'nur_rueckbau', False) or             getattr(args, 'elementbudget', None):
        if getattr(args, 'rueckbau_faktor', None):
            opts += ['--prune-factor', repr(args.rueckbau_faktor)]
        if getattr(args, 'rueckbau_versuche', None):
            opts += ['--prune-attempts', str(args.rueckbau_versuche)]
        if getattr(args, 'rueckbau_stufen', None):
            opts += ['--prune-stages', str(args.rueckbau_stufen)]
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


# Einstellungen des erzwungenen Einfügens (M6-T6b: --erzwungen-min-spalt, --erzwungen-blende);
# Standard wie im dokumentierten Lauf M6-T6
FORCED = {'min_gap': 0.7, 'stop_gap': False}


def forced_insertion(runner, tag, src, relax_options, glass='SYN-36', thickness=0.4,
                     min_gap=None, curvature_bound=1.2, log=print):
    """Erzwungenes Einfügen, wenn der Controller ohne aktive Geisterplatte stehen bleibt.

    Für jeden Luftspalt ohne Geisterplatten (außer dem Blendenspalt, mit FORCED['stop_gap'] auch
    dort) mit Dicke >= min_gap (Standard FORCED['min_gap']) wird
    mit `mikroptim insert` eine Planplatte (c₀ = 0) der Dicke `thickness` aus `glass` in die
    Mitte gesetzt (Krümmungen variabel in ±curvature_bound, Dicke variabel), danach alle
    Kandidaten nachoptimiert; der mit der kleinsten Merit gewinnt. Liefert (Datei, Liste der
    Kandidaten) oder (None, []) ohne passenden Spalt.
    """
    s = load(src)
    gaps = plate_gaps(s)
    if min_gap is None:
        min_gap = FORCED['min_gap']
    cands = []
    for i in range(0 if FORCED['stop_gap'] else 1, len(s['surfaces']) - 1):
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


def limit_elements(opts, n):
    """Höchste Elementzahl n anhängen; mit Elementbudget (M6-T9, `--element-budget`, schließt
    `--max-elements` aus) stattdessen das Budget auf min(Budget, n) setzen."""
    opts = list(opts)
    if '--element-budget' in opts:
        k = opts.index('--element-budget') + 1
        opts[k] = str(min(int(opts[k]), n))
        return opts
    return opts + ['--max-elements', str(n)]


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
            opts = limit_elements(opts, glass_elements(load(cur)) + element_budget)
        if stage_limit is not None:
            opts = limit_elements(opts, stage_limit)
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
    Glas-Hülle (falls Glasvariablen) in drei Stufen ×10 bis zum Gewicht der Datei. Mit
    spectrum=None bleibt das Spektrum der Datei (M6-T10)."""
    s = load(src)
    strip_ghost_plates(s)
    remove_dummy_planes(s)
    buchdahl_to_catalog(s)
    if spectrum is not None:
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


def psf_numerical_aperture(system):
    """NA für `mikroptim eval --psf --na` aus der Datei: kollimierte Quelle mit Pupillenradius
    a = f · NA (Konvention von synth), also NA = a / f mit f = target.focal_length."""
    f = system.get('target', {}).get('focal_length', FOCAL)
    return system['surfaces'][0]['semi_diameter'] / f


def evaluate(runner, name, src, grid=256):
    """mikroptim eval: Raster mit Referenz durch die Austrittspupille und durch den Scheitel
    (bzw. mit unendlichem Radius wie die Merit, wenn die Datei `image_infinite` hat), skalare und
    vektorielle PSF (JSON).

    Kann eval die NA der Pupillenabbildung nicht bestimmen (ein Feldstrahl versagt, etwa bei
    vollem Feld eines bei 10 % Feld entworfenen Systems: „NA vorgeben“), wird die PSF mit der NA
    der Datei (`psf_numerical_aperture`) wiederholt, statt den Lauf abzubrechen (Fehler aus
    M6-T7); die Ausgabe nennt die Variante dann in `<name>_<key>_na.txt`.
    """
    sub = 'e' + 'val'
    system = load(src)
    second = (['--reference', 'infinite'] if system.get('reference_sphere') == 'image_infinite'
              else ['--reference', 'vertex'])
    variants = {
        'pupille': ['--reference', 'exit-pupil'],
        'unendlich' if second[1] == 'infinite' else 'scheitel': second,
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
            err_path = runner.path(f'{name}_{key}_stderr.txt')
            with open(js, 'w', encoding='utf-8') as f, open(err_path, 'w', encoding='utf-8') as e:
                rc = subprocess.run(args, stdout=f, stderr=e, cwd=ROOT).returncode
            if rc != 0 and key in ('psf', 'vektor'):
                with open(err_path, encoding='utf-8', errors='replace') as e:
                    needs_na = 'NA vorgeben' in e.read()
                if needs_na:
                    na = psf_numerical_aperture(system)
                    with open(js, 'w', encoding='utf-8') as f, \
                            open(err_path, 'w', encoding='utf-8') as e:
                        rc = subprocess.run(args + ['--na', repr(na)], stdout=f, stderr=e,
                                            cwd=ROOT).returncode
                    with open(runner.path(f'{name}_{key}_na.txt'), 'w', encoding='utf-8') as f:
                        f.write(f'NA der Pupillenabbildung nicht bestimmbar; --na {na!r} '
                                f'(Datei: Pupillenradius / Brennweite)\n')
                    print(f'{name}_{key}: NA nicht bestimmbar, wiederholt mit --na {na:.4f}',
                          flush=True)
            runner.times[f'{name}_{key}'] = time.time() - t0
            save(runner.times, runner.times_file)
            print(f'{name}_{key}: Exit {rc}', flush=True)
            if rc != 0:
                os.replace(js, js + '.fehler')  # kein unvollständiges JSON für die Fortsetzung
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
                second = 'unendlich' if 'unendlich' in ev else 'scheitel'
                txt, worst = tuple_table(ev[second])
                label = 'unendlich, wie die Merit' if second == 'unendlich' else 'Scheitel'
                print(f'\n## {key} (Referenz {label}), größter RMS {fmt(worst)}\n')
                print(txt)
                print()
        final = load(p8['final'])
        print('## Flächen des Endsystems\n')
        print(surface_table(final))


def set_reference_sphere(system, kind):
    """Feld reference_sphere der Startdatei (M6-T7, ADR-0017). Auch `last_surface` wird
    ausdrücklich gesetzt: bitgleich zu einer Datei ohne Feld, aber ohne die Warnung der CLI."""
    system['reference_sphere'] = kind


def f12_guard(args):
    """Hilfsbedingung paraxial_focus gegen die Falle F12 im OSC-Versuch: nur bei der alten
    Konvention `last_surface` standardmäßig an (M6-T7: mit der neuen Konvention entbehrlich)."""
    if args.mit_f12_schutz:
        return True
    return args.referenzsphaere == 'last_surface' and not args.ohne_f12_schutz


def aperture_branch(args):
    """Vergleichszweig: Apertur zuerst (Standardfolge von optimierung.md §7) bei 10 % Feld."""
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.out), gpu=args.gpu,
               synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    s = load(args.start)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    set_reference_sphere(s, args.referenzsphaere)  # M6-T7 (ADR-0017)
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
    summary = {'start': args.start, 'reference_sphere': args.referenzsphaere,
               'rounds': ra, 'point': pt, 'final': a2,
               'eval': evaluate(r, 'a3', a2), 'eval_full_field': evaluate(r, 'a3_vollfeld', full),
               'times': r.times, 'seconds_total': time.time() - t0}
    with open(r.path('m6t6_apertur_zweig.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print('fertig:', a2)
    return 0


def osc_weight_for_field(fraction, na=0.95, wavelength_nm=546.07, zones=2):
    """Gewicht des Residuums sine_condition, bei dem √w·OSC in allen Zonen zusammen so viel kostet
    wie die lineare Koma beim Feldanteil `fraction` (docs/optimierung.md, Abschnitt 2.7):
    W131 = OSC · NA · h / λ (h Bildhöhe), RMS der Koma ohne Kippung W131/√72, also
    w = (NA h / λ)² / (72 · Zonen)."""
    h = FOCAL * math.tan(math.radians(fraction * FIELD_DEG))
    return (na * h / (wavelength_nm * 1e-6)) ** 2 / (72.0 * zones)


def osc_evaluate(r, name, src):
    """Bewertung bei 0, 7 und 10 % Feld (Winkel der Feldstufe des Hauptlaufs; Referenz
    Austrittspupille) und OSC in Zone und Rand aus `optimize --max-iterations 0
    --sine-condition 0` (Batch, mit Geisterplatten)."""
    s = load(src)
    set_field(s, [a * 0.1 for a in FIELD_ANGLES], FIELD_WEIGHTS)
    ev_in = r.path(name + '_feld_in.json')
    save(s, ev_in)
    ev = evaluate(r, name, ev_in)
    rms = []
    for t in load(ev['pupille'])['tuples']:
        lam = t['wavelength_nm'] * 1e-6
        rms.append({'field_deg': t['field_angle_deg'],
                    'rms_waves': t['rms_opd_mm'] / lam if t['rms_opd_mm'] is not None else None,
                    'rms_without_tilt_waves': t['rms_opd_tilt_waves'],
                    'strehl_marechal': t['strehl_marechal'], 'failed_rays': t['failed_rays'],
                    'distortion_percent': t['distortion_chief_percent']})
    o = subprocess.run([r.exe, 'optimize', src, '--max-iterations', '0', '--sine-condition', '0',
                        '--json', '-q', '--rings', '16'], capture_output=True, text=True,
                       encoding='utf-8', cwd=ROOT)
    osc = {}
    if o.returncode == 0:
        for c in json.loads(o.stdout)['constraints']:
            if c['type'] == 'sine_condition':
                osc[repr(c['zone'])] = c['value_start']
    return {'eval': ev, 'rms': rms, 'osc': osc}


def osc_branch(args):
    """M11-T4: Versuch „Apertur zuerst auf der Achse“ (Befund F4) mit und ohne Residuum der
    Sinusbedingung. NA 0,3 -> 0,95 nur auf der Achse (546 nm, Nebenbedingungen des Startsystems;
    mit --osc-strahlsuche Strahlsuche B = 2 mit 2 Varianten wie der Vergleichszweig). Bewertet
    bei 0, 7 und 10 % Feld: der Endpunkt der Synthese (mit Geisterplatten) und das System nach
    Entfernen der Geisterplatten und Hilfsebenen und Nachoptimierung auf der Achse (2000
    Zyklen, Residuum wie in der Synthese). Gewicht 0: ohne Residuum."""
    w = args.osc_gewicht
    tag = ('mit' if w > 0 else 'ohne') + ('_strahlsuche' if args.osc_strahlsuche else '')
    guard = f12_guard(args)
    if args.referenzsphaere != 'last_surface':
        tag += '_' + args.referenzsphaere
    if not guard:
        tag += '_ohne_f12'
    out = os.path.join(os.path.abspath(args.out), tag)
    r = Runner(os.path.abspath(args.exe), out, gpu=args.gpu, synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    s = load(args.start)
    set_field(s, [0.0], [1.0])
    # M6-T7 (ADR-0017): Referenzsphäre unabhängig von der letzten Fläche
    set_reference_sphere(s, args.referenzsphaere)
    # Gegen die Falle F12 (Fokus im Scheitel der Hilfsebene vor dem Objekt, Merit klein, reales
    # System defokussiert): Bildebene im paraxialen Fokus, schwach (0,1 mm Randstrahlhöhe kostet
    # eine Welle², die Falle mit 2,7 mm Defokus rund 700 Wellen²; die Fokuslage für den Ausgleich
    # der Restsphärik, einige µm, bleibt praktisch frei)
    if guard:
        set_constraint(s, {'type': 'paraxial_focus', 'weight': 100.0,
                           'comment': 'M11-T4-Versuch: gegen F12 (Fokus in der Hilfsebene)'})
    if w > 0:
        set_constraint(s, {'type': 'sine_condition', 'weight': w,
                           'comment': 'M11-T4: OSC in Zone 1/√2 und am Rand, Gewicht wie die '
                                      'Koma bei 10 % Feld'})
    start = r.path('o0_start.json')
    save(s, start)
    t0 = time.time()
    beam = ['--beam', '2', '--alternatives', '2'] if args.osc_strahlsuche else []
    o1, ro = synth_loop(r, 'o1', start,
                        ['--target-na', '0.95', '--curvature-bound', '1.2', '--max-steps', '3000',
                         '--max-iterations', '500', '--max-elements', '14'] + beam + rings,
                        relax, lambda p: p['aperture'], 0.95, max_rounds=12, forced=2)
    pt = last_point(r, ro)
    endpoint = osc_evaluate(r, 'o1e', o1)
    s = load(o1)
    strip_ghost_plates(s)
    remove_dummy_planes(s)
    buchdahl_to_catalog(s)
    add_gap_variables(s)
    prep = r.path('o2_in.json')
    save(s, prep)
    o2 = r.optimize('o2', prep, ['--max-iterations', '2000', '-q'] + rings)
    polished = osc_evaluate(r, 'o3', o2)
    summary = {'start': args.start, 'osc_weight': w, 'beam': bool(beam),
               'reference_sphere': args.referenzsphaere,
               'f12_guard': guard, 'rounds': ro,
               'point': pt, 'synth_result': o1, 'elements_synth': glass_elements(load(o1)),
               'endpoint': endpoint, 'final': o2, 'elements': glass_elements(load(o2)),
               'polished': polished, 'times': r.times, 'seconds_total': time.time() - t0}
    with open(r.path('m11t4_osc.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    for key, e in (('Endpunkt', endpoint), ('nachoptimiert', polished)):
        print(f"{key}: NA {pt['aperture']}, RMS "
              + ', '.join(f"{x['rms_waves']:.4f}" for x in e['rms'] if x['rms_waves'] is not None)
              + f", OSC {e['osc']}")
    return 0


# ---------------------------------------------------------------------------------------------
# M6-T10: Rückkehr zu Geisterplatten und zyklischer Ablauf
# ---------------------------------------------------------------------------------------------

# Glas-Wörterbuch der Rückkehr: die Gläser des Startsystems (dort als Buchdahl-Fits der frei
# erfundenen Testgläser, hier mit Katalognamen wie nach buchdahl_to_catalog)
GHOST_GLASSES = ['SYN-36', 'CAF2', 'SYN-30']


def first_json(text):
    """Erstes JSON-Objekt eines Textes (Warnungen auf stderr landen in derselben Datei)."""
    i = text.index('{')
    return json.JSONDecoder().raw_decode(text[i:])[0]


def return_to_ghosts(runner, name, src, positions='aperture,field', glasses=None, order=10,
                     weight=None, norm_field=1.0):
    """Rückkehr vom realen System zu Geisterplatten (`mikroptim insert --ghost-return`, M6-T10):
    Wörterbücher mit Koeffizienten 0 in der Mitte der Luftspalte der Lagen `positions`, diese
    Spalte fest, Normierungsradien aus den realen Strahlhöhen bei Feldwinkeln × norm_field.
    Merit unverändert (Bericht: merit_before/merit_after). Liefert (Datei, Bericht)."""
    out = runner.path(name + '.json')
    text = runner.path(name + '_bericht.txt')
    if not os.path.exists(out):
        a = ['insert', src, '-o', out, '--ghost-return', '--ghost-positions', positions,
             '--ghost-order', str(order), '--norm-field', repr(norm_field), '--json']
        for g in glasses or GHOST_GLASSES:
            a += ['--ghost-glass', g]
        if weight is not None:
            a += ['--ghost-weight', repr(weight)]
        runner.run(name, a, text_name=name + '_bericht.txt')
    with open(text, encoding='utf-8') as f:
        return out, first_json(f.read())


def system_state(system):
    """Punkt eines realen Systems: NA (Pupillenradius / f), Feldanteil (größter Feldwinkel /
    Feld der Spezifikation), Spektrum (Schlüssel von SPECTRA oder None), Glaselemente."""
    angles = system['source'].get('field_angles_deg') or [0.0]
    wl = list(system['spectrum']['wavelengths_nm'])
    key = next((k for k, (w, _) in SPECTRA.items() if list(w) == wl), None)
    return {'aperture': psf_numerical_aperture(system), 'field': max(angles) / FIELD_DEG,
            'spectrum': key, 'elements': glass_elements(system)}


def stage_done(kind, state):
    if kind == 'feld':
        return state['field'] >= 0.999
    if kind == 'apertur':
        return state['aperture'] >= 0.95 - 1e-4
    return state['spectrum'] in ('voll', 'erweitert')


def prune_point(system, kind, q, na, field):
    """Punkt des Rückbaus bei Q × Parameter der Stufe (M6-T11, Vorschlag aus M6-T10/F17): Apertur
    Q·NA (Pupillenradius der Datei), Feld Q·Feldanteil, Farbe Spektrumbreite Q (Wellenlängen um
    λ_ref skaliert wie die stetige Spektrum-Homotopie). Ändert `system`; liefert (NA, Feld)."""
    if kind == 'apertur':
        f = system.get('target', {}).get('focal_length', FOCAL)
        system['surfaces'][0]['semi_diameter'] = f * q * na
        return q * na, field
    if kind == 'feld':
        return na, q * field
    sp = system['spectrum']
    ref = sp['reference_nm']
    sp['wavelengths_nm'] = [ref + q * (w - ref) for w in sp['wavelengths_nm']]
    return na, field


def reopen_after_prune(r, tag, pruned, kind, na, field, state, common, zopt):
    """Nach dem Rückbau bei Q × Parameter zurück zum erreichten Punkt (M6-T11): Apertur und Feld
    mit synth ohne Einfügen (höchstens die Elemente des Rückbaus), Farbe direkt über die
    Nachoptimierung Z2 am vollen Spektrum. Erreicht synth den Punkt nicht, bleibt der Stand vor
    dem Rückbau (Rückgabe None statt der Datei). Liefert (Datei oder None, Bericht)."""
    if kind == 'farbe':
        return pruned, {'reached': True, 'via': 'z2'}
    s = load(pruned)
    n = glass_elements(s)
    p0 = system_state(s)
    if kind == 'apertur':
        opts = ['--target-na', repr(na), '--start-field', repr(field), '--target-field',
                repr(field)]
    else:
        opts = ['--target-na', repr(p0['aperture']), '--start-field', repr(load(
            r.path(tag + '_rueckbau_log.json'))['result']['point']['field']),
                '--target-field', repr(field)]
    out = r.synth(tag + '_wieder', pruned, opts + limit_elements(common, n))
    res = load(r.path(tag + '_wieder_log.json'))['result']
    pt = res['point']
    reached = (pt['aperture'] >= na - 1e-9) and (pt['field'] >= field - 1e-9)
    info = {'reached': reached, 'status': res['status'], 'point': pt, 'elements': n}
    return (out if reached else None), info


def cycle_stage(r, args, k, kind, cur, common, relax, rings, log=print):
    """Eine Stufe des Zyklus (M6-T10): reales System → Geisterplatten (Rückkehr) → Stufe
    (synth_loop wie im Hauptlauf) → Z2 (ohne Platten nachoptimiert) → Rückbau am realen System
    (synth --prune am erreichten Punkt) → Z2 nach dem Rückbau, bewertet."""
    st = system_state(load(cur))
    na, f = st['aperture'], st['field']
    tag = f'c{k}_{kind}'
    log(f"Stufe {tag}: NA {na:.4f}, Feld {f:.3f}, Spektrum {st['spectrum']}, "
        f"{st['elements']} Elemente")
    t0 = time.time()
    s = load(cur)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    real = r.path(tag + '_real.json')
    save(s, real)
    start, report = return_to_ghosts(r, tag + '_start', real, positions=args.geister_lagen)
    log(f"  Rückkehr: {len(load(start).get('ghost_plates', []))} Platten, F "
        f"{report['merit_before']:.6e} -> {report['merit_after']:.6e}")
    info = {'kind': kind, 'tag': tag, 'start': cur, 'start_state': st,
            'ghost_return': report, 'ghost_start': start}

    def full_field(system, point):
        set_field(system, FIELD_ANGLES, FIELD_WEIGHTS)

    rounds = []
    if kind == 'feld':
        def opts(point):
            f0 = f if point is None else point['field']
            return ['--target-na', repr(na), '--start-field', repr(f0), '--target-field',
                    '1'] + common
        res, rounds = synth_loop(r, tag, start, opts, relax, lambda p: p['field'], 1.0,
                                 restart=full_field, max_rounds=args.zyklus_runden, forced=2,
                                 log=log)
    elif kind == 'apertur':
        res, rounds = synth_loop(r, tag, start,
                                 ['--target-na', '0.95', '--start-field', repr(f),
                                  '--target-field', repr(f)] + common, relax,
                                 lambda p: p['aperture'], 0.95, restart=full_field,
                                 max_rounds=args.zyklus_runden, forced=3, log=log)
    else:
        seq = {'mono': ['zwei', 'voll'], None: ['zwei', 'voll'], 'zwei': ['voll']}[
            st['spectrum']]
        if getattr(args, 'farb_ziel', None) == 'voll':
            seq = ['voll']  # M6-T11: direkt zum Zielspektrum (mit --farb-stetig weich ab λ_ref)
        res = start
        stufen = (['--spectrum-continuous'] if args.farb_stetig else
                  ['--spectrum-stages', str(args.farb_stufen)] if args.farb_stufen > 0 else [])
        for sub in seq:
            s = load(res)
            wl, w = SPECTRA[sub]
            set_spectrum(s, wl, w)
            inp = r.path(f'{tag}_{sub}_in.json')
            save(s, inp)

            def full_field_spectrum(system, point, wl=wl, w=w):
                # M6-T11: synth schreibt das Spektrum des Endpunkts (Spektrumstufe bzw. -breite);
                # der Neustart braucht wieder das Zielspektrum der Teilstufe
                set_field(system, FIELD_ANGLES, FIELD_WEIGHTS)
                set_spectrum(system, wl, w)

            restart = full_field_spectrum if (args.farb_stetig or args.farb_stufen > 0) else \
                full_field
            res, rr = synth_loop(r, f'{tag}_{sub}', inp,
                                 ['--target-na', repr(na), '--start-field', repr(f),
                                  '--target-field', repr(f)] + stufen + common, relax,
                                 (lambda p: p.get('spectrum_width', 0.0)) if args.farb_stetig
                                 else (lambda p: 0.0), 1.0, restart=restart, max_rounds=4, forced=1,
                                 forced_kind='split', stage_budget=args.farb_budget, log=log)
            rounds += rr
    pt = rounds[-1]['point']
    fk, nak = pt['field'], pt['aperture']
    info.update({'result': res, 'rounds': rounds, 'point': pt,
                 'elements_stage': glass_elements(load(res))})
    # Z2: ohne Geisterplatten am erreichten Punkt nachoptimiert (Spektrum der Stufe)
    zopt = ['--max-iterations', '2000', '-q'] + rings
    s = load(res)
    set_field(s, [a * fk for a in FIELD_ANGLES], FIELD_WEIGHTS)
    z2_in = r.path(tag + '_z2_punkt.json')
    save(s, z2_in)
    z2 = polish(r, tag + '_z2', z2_in, zopt, spectrum=None)
    # Rückbau am realen System (M6-T9: dort wirkt er, F17), Punkt = erreichter Punkt bzw. mit
    # --rueckbau-punkt Q (M6-T11) Q × Parameter der Stufe (Abstand zur Schwelle, F17)
    q = getattr(args, 'rueckbau_punkt', None)
    s = load(z2)
    set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
    nap, fp = nak, fk
    if q:
        nap, fp = prune_point(s, kind, q, nak, fk)
    p_in = r.path(tag + '_rueckbau_in.json')
    save(s, p_in)
    prune_opts = ['--prune', '--prune-factor', repr(args.rueckbau_faktor or 0.5),
                  '--prune-attempts', str(args.rueckbau_versuche or 30)]
    if args.rueckbau_stufen:
        prune_opts += ['--prune-stages', str(args.rueckbau_stufen)]
    # höchstens die Elemente von Z2: ist der Startpunkt nicht annehmbar (F_B > F_max), fügte
    # synth sonst zuerst ein (Lauf D: 13 → 14 Elemente statt Rückbau)
    pruned = r.synth(tag + '_rueckbau', p_in,
                     ['--target-na', repr(nap), '--start-field', repr(fp), '--target-field',
                      repr(fp)] + prune_opts +
                     limit_elements(common, glass_elements(load(z2))))
    pres = load(r.path(tag + '_rueckbau_log.json'))['result']
    reopen = None
    if q and pres.get('removals', 0) > 0:
        # zurück zum erreichten Punkt ohne Einfügen; erreicht der Lauf ihn nicht, bleibt Z2
        back, reopen = reopen_after_prune(r, tag, pruned, kind, nak, fk, st, common, zopt)
        pruned = back if back is not None else z2
    elif q:
        pruned, reopen = z2, {'reached': True, 'skipped': 'nichts entfernt'}
    s = load(pruned)
    set_field(s, [a * fk for a in FIELD_ANGLES], FIELD_WEIGHTS)
    if q:
        s['spectrum'] = copy.deepcopy(load(z2)['spectrum'])  # Farbstufe: Spektrum zurück
    z2r_in = r.path(tag + '_z2r_punkt.json')
    save(s, z2r_in)
    z2r = polish(r, tag + '_z2r', z2r_in, zopt, spectrum=None)
    info.update({'z2': z2, 'elements_z2': glass_elements(load(z2)),
                 'prune': {'status': pres['status'], 'removals': pres.get('removals', 0),
                           'attempts': pres.get('removal_attempts', 0),
                           'elements_before': pres.get('start_elements'),
                           'elements_after': pres['elements'],
                           'point': {'aperture': nap, 'field': fp,
                                     'spectrum_width': q if (q and kind == 'farbe') else 1.0},
                           'reopen': reopen},
                 'z2r': z2r, 'elements': glass_elements(load(z2r)),
                 'eval': evaluate(r, tag + '_z2r', z2r)})
    if fk < 0.999:
        s = load(z2r)
        set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
        full = r.path(tag + '_z2r_vollfeld_in.json')
        save(s, full)
        info['eval_full_field'] = evaluate(r, tag + '_z2r_vollfeld', full)
    info['end_state'] = system_state(load(z2r))
    info['seconds'] = time.time() - t0
    log(f"  {tag}: Punkt NA {nak:.4f}, Feld {fk:.3f}; {info['elements_stage']} Elemente, "
        f"nach dem Rückbau {info['elements']}")
    return info


def cycle_run(args):
    """Zyklischer Ablauf (M6-T10) ab einem realen System (z. B. Z2 nach dem Rückbau aus M6-T9):
    für jede Stufe aus --zyklus (feld, farbe, apertur) Rückkehr zu Geisterplatten, Stufe, Z2,
    Rückbau am realen System, Z2 danach; die nächste Stufe beginnt am verkleinerten System.
    Stufen am Ziel werden übersprungen; der Durchlauf wiederholt sich (--zyklus-wiederholen),
    bis alle am Ziel sind oder ein Durchlauf nichts mehr öffnet."""
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.out), gpu=args.gpu,
               synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    common = ['--curvature-bound', '1.2', '--max-steps', '3000', '--max-iterations', '500'] + \
        (['--element-budget', str(args.elementbudget)] if args.elementbudget else
         ['--max-elements', '24']) + rings
    s = load(args.start)
    set_reference_sphere(s, args.referenzsphaere)
    cur = r.path('c_start.json')
    save(s, cur)
    kinds = args.zyklus.split(',')
    for kind in kinds:
        if kind not in ('feld', 'farbe', 'apertur'):
            raise SystemExit(f'--zyklus: unbekannte Stufe {kind} (feld, farbe, apertur)')
    summary = {'start': args.start, 'start_state': system_state(s), 'kinds': kinds,
               'synth_options': r.synth_options, 'ghost_positions': args.geister_lagen,
               'element_budget': args.elementbudget, 'stages': []}
    # M6-T11: nur mit den Optionen (Zusammenfassungen von M6-T10 bleiben gleich)
    if getattr(args, 'rueckbau_punkt', None):
        summary['prune_point'] = args.rueckbau_punkt
    if getattr(args, 'farb_ziel', 'zwei') != 'zwei':
        summary['color_target'] = args.farb_ziel
    path = r.path('m6t10_zyklus.json')

    def write():
        summary['times'] = r.times
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)

    t0 = time.time()
    k = 0
    # Stufen ohne Fortschritt nicht am selben Zustand wiederholen (erst wieder, wenn eine andere
    # Stufe etwas geöffnet hat)
    stalled = set()
    for _ in range(args.zyklus_wiederholen):
        moved = False
        for kind in kinds:
            st = system_state(load(cur))
            if stage_done(kind, st) or kind in stalled:
                continue
            info = cycle_stage(r, args, k, kind, cur, common, relax, rings,
                               log=lambda m: print(m, flush=True))
            summary['stages'].append(info)
            write()
            new = info['end_state']
            if (new['field'] > st['field'] + 1e-3 or new['aperture'] > st['aperture'] + 1e-3 or
                    new['spectrum'] != st['spectrum']):
                moved = True
                stalled = set()
            else:
                stalled.add(kind)
            cur = info['z2r']
            k += 1
        if all(stage_done(kind, system_state(load(cur))) for kind in kinds) or not moved:
            break
    summary['final'] = cur
    summary['final_state'] = system_state(load(cur))
    summary['seconds_total'] = time.time() - t0
    write()
    print('fertig:', cur)
    return 0


def rooted(path):
    """Pfad aus einer Zusammenfassung (absolut oder nach --archiv relativ zur Wurzel)."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def eval_rows(pupil_json, psf_json=None):
    """RMS [Wellen bei λ] und Strehl (FFT) pro (Feldwinkel, λ) aus den Bewertungen."""
    e = load(pupil_json)
    psf = load(psf_json)['tuples'] if psf_json and os.path.exists(psf_json) else None
    out = []
    for k, t in enumerate(e['tuples']):
        lam = t['wavelength_nm'] * 1e-6
        rms = t['rms_opd_mm'] / lam if t['rms_opd_mm'] is not None else None
        s = psf[k]['psf']['strehl'] if psf and psf[k].get('psf') else None
        out.append((t['field_angle_deg'], t['wavelength_nm'], rms, s, t['failed_rays']))
    return out


def cycle_tables(out_dir):
    """Tabelle der Stufen eines zyklischen Laufs (Markdown auf stdout): Punkt, Spektrum,
    Elemente (Stufe, nach dem Rückbau), größter RMS über λ bei 0/70/100 % des Feldanteils,
    kleinster Strehl, Laufzeit."""
    run = load(os.path.join(out_dir, 'm6t10_zyklus.json'))
    times = run.get('times', {})
    print('| Stufe | NA | Feld | Spektrum | El. Stufe → Z2 → Rückbau | RMS 0/70/100 % (max über λ) '
          '| Strehl min (FFT) | Rückkehr ΔF/F | Zeit [s] |')
    print('|---|---|---|---|---|---|---|---|---|')
    for st in run['stages']:
        rows = eval_rows(rooted(st['eval']['pupille']), rooted(st['eval'].get('psf')))
        fields = sorted({x[0] for x in rows})
        rms = [max(x[2] for x in rows if x[0] == fa and x[2] is not None) for fa in fields]
        strehl = [x[3] for x in rows if x[3] is not None]
        e = st['end_state']
        t = sum(v for key, v in times.items() if key.startswith(st['tag'] + '_'))
        print(f"| {st['tag']} | {fmt(e['aperture'], 4)} | {fmt(e['field'], 3)} | "
              f"{e['spectrum']} | {st['elements_stage']} → {st['elements_z2']} → "
              f"{st['elements']} | {'/'.join(fmt(x, 3) for x in rms)} | "
              f"{fmt(min(strehl), 3) if strehl else '–'} | "
              f"{st['ghost_return']['relative_change']:.1e} | {t:.0f} |")
    print(f"\nGesamt: {sum(times.values()):.0f} s")
    last = run['stages'][-1] if run['stages'] else None
    if last:
        print(f"\n## Tupel von {last['tag']}_z2r\n")
        print('| Feld [°] | λ [nm] | RMS [Wellen bei λ] | Strehl (FFT) | versagt |')
        print('|---|---|---|---|---|')
        for fa, wl, rms, s, failed in eval_rows(rooted(last['eval']['pupille']),
                                                rooted(last['eval'].get('psf'))):
            print(f'| {fmt(fa, 3)} | {fmt(wl, 2)} | {fmt(rms)} | {fmt(s, 3)} | {failed} |')


def archive(out_dir):
    """Ausgabeverzeichnis für das Repository verkleinern: behalten werden die Zusammenfassungen,
    die Protokolle der synth-Läufe (Text und JSON), die Ergebnisdateien jeder Stufe, die Texte
    der Feinoptimierung und des Einrastens sowie die Bewertungen; absolute Pfade werden relativ
    zum Wurzelverzeichnis geschrieben. Die übrigen Zwischenstände (Nachoptimierungen, Kandidaten
    der erzwungenen Einfügungen) entstehen beim Wiederholen des Laufs neu."""
    import re
    keep = re.compile(r'^(m6t6_.*\.json|[pa]\d+_r\d+_log\.json|[pa]\d+_r\d+\.txt|'
                      r'p0_start\.json|a0_start\.json|p6_s\d\.txt|p6_s2\.json|'
                      r'p7\.(json|txt)|z2_s0\.(json|txt)|a2\.(json|txt)|lauf\.log|'
                      r'(m\d)_s0\.(json|txt)|.*_na\.txt|'
                      r'(p8|z2|a3|m\d)(_vollfeld)?_(pupille|scheitel|unendlich|psf|vektor)\.json|'
                      # M6-T10: zyklischer Lauf
                      r'm6t10_zyklus\.json|c_start\.json|c\d+_[a-z]+_start(\.json|_bericht\.txt)|'
                      r'c\d+_[a-z]+(_zwei|_voll)?_r\d+(_log\.json|\.txt)|'
                      r'c\d+_[a-z]+_(rueckbau|wieder)(_log\.json|\.txt)|c\d+_[a-z]+_z2r?_s\d\.(json|txt)|'
                      r'c\d+_[a-z]+_z2r(_vollfeld)?_(pupille|unendlich|psf|vektor)\.json)$')
    # Ergebnisdateien der Stufen (auch Neustart-Dateien, wenn eine Stufe dort endete)
    referenced = set()
    for summary in ('m6t6_lauf.json', 'm6t6_apertur_zweig.json', 'm6t10_zyklus.json'):
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
        with open(path, 'rb') as f:
            raw = f.read()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            # lauf.log unter Windows in der ANSI-Codepage der Konsole (M6-T8)
            text = raw.decode('cp1252')
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
    if len(sys.argv) > 1 and sys.argv[1] == '--zyklus-tabelle':
        cycle_tables(sys.argv[2])
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
    ap.add_argument('--farbe-von-anfang', choices=['zwei', 'voll'], default=None,
                    help='Spektrum ab dem Start (ohne eigene Farbstufen); mit --farbregel')
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
    ap.add_argument('--osc-versuch', action='store_true',
                    help='M11-T4: Apertur zuerst nur auf der Achse (NA 0,3 -> 0,95), Bewertung bei '
                         '0/7/10 %% Feld; mit --osc-gewicht W > 0 mit Residuum sine_condition, '
                         'Ausgabe in OUT/mit bzw. OUT/ohne')
    ap.add_argument('--osc-strahlsuche', action='store_true',
                    help='OSC-Versuch mit Strahlsuche (synth --beam 2 --alternatives 2)')
    ap.add_argument('--referenzsphaere', default='image_infinite',
                    choices=['last_surface', 'image_exit_pupil', 'image_infinite'],
                    help='M6-T7/M6-T6b: Referenzsphäre der Merit (Feld reference_sphere der '
                         'Startdatei; Standard image_infinite, ADR-0017). Die Läufe vor M6-T7 '
                         '(M6-T6, M11-T4) mit --referenzsphaere last_surface nachrechnen')
    ap.add_argument('--ohne-f12-schutz', action='store_true',
                    help='OSC-Versuch ohne die Hilfsbedingung paraxial_focus gegen die Falle F12 '
                         '(nur bei last_surface nötig; mit der neuen Konvention Standard)')
    ap.add_argument('--mit-f12-schutz', action='store_true',
                    help='OSC-Versuch mit paraxial_focus auch bei der neuen Konvention (M6-T7, E)')
    ap.add_argument('--erzwungen-min-spalt', type=float, default=0.7, metavar='G',
                    help='M6-T6b: kleinster Luftspalt [mm] für das erzwungene Einfügen (Standard 0,7)')
    ap.add_argument('--erzwungen-blende', action='store_true',
                    help='M6-T6b: erzwungenes Einfügen auch im Blendenspalt')
    ap.add_argument('--erzwungen-mehr', type=int, default=0, metavar='N',
                    help='M6-T6b: N erzwungene Einfügungen mehr in den Stufen 1–3 (Standard 0)')
    ap.add_argument('--sinusbedingung', type=float, default=None, metavar='W',
                    help='M6-T6b: synth --sine-condition W in allen Stufen (Residuum der '
                         'Sinusbedingung, M11-T4); Standard aus')
    ap.add_argument('--gemeinsam', action='store_true',
                    help='M6-T8: Stufen 1 und 2 als eine gemeinsame Apertur-/Feld-Homotopie '
                         '(synth --joint-aperture-field): NA vom Start bis --na-feld und Feld '
                         '10 -> 100 %% zugleich')
    ap.add_argument('--gemeinsam-exponent', type=float, default=None, metavar='P',
                    help='M6-T8: Feld(t) = 0,1 + t^P · 0,9 (synth --joint-field-exponent; '
                         'Standard 1: Diagonale)')
    ap.add_argument('--feldregel', action='store_true',
                    help='M6-T8: Feldregel in allen synth-Läufen (synth --field-rule)')
    ap.add_argument('--einfuegen-verbessern', type=int, default=None, metavar='N',
                    help='M6-T8: Einfügen nur mit Verbesserung von F_B, bis N Varianten '
                         '(synth --insertion-improve N)')
    ap.add_argument('--ab-p2', action='store_true',
                    help='M6-T8: Stufe 1 überspringen, die Startdatei ist ihr Ergebnis (z. B. '
                         'p1_final.json von M6-T6b V2 bei NA 0,855); Feldstufe bei deren NA')
    ap.add_argument('--bis', choices=['p2'], default=None,
                    help='M6-T8: nach Stufe 2 (Feldstufe) und dem Zwischenkandidaten Z2 enden')
    ap.add_argument('--rueckbau', action='store_true',
                    help='M6-T9: Rückbau in allen synth-Läufen (synth --prune)')
    ap.add_argument('--rueckbau-faktor', type=float, default=None, metavar='F',
                    help='M6-T9: Rückbau nur bei F_B <= F x F_max (synth --prune-factor)')
    ap.add_argument('--rueckbau-versuche', type=int, default=None, metavar='N',
                    help='M6-T9: Kandidaten pro Rückbau-Runde (synth --prune-attempts)')
    ap.add_argument('--rueckbau-stufen', type=int, default=None, metavar='K',
                    help='M6-T9: Stufen der Rückbau-Homotopie (synth --prune-stages)')
    ap.add_argument('--elementbudget', type=int, default=None, metavar='N',
                    help='M6-T9: höchstens N Glaselemente pro synth-Lauf statt 24, vorher '
                         'Rückbau (synth --element-budget N)')
    ap.add_argument('--nur-rueckbau', action='store_true',
                    help='M6-T9: Startdatei ist ein Stand bei vollem Feld (z. B. Ende der '
                         'Feldstufe oder Z2); ein synth-Lauf mit Rückbau an ihrem Punkt '
                         '(NA der Datei, Feldanteil --rueckbau-feld), danach Z2 wie --bis p2')
    ap.add_argument('--rueckbau-feld', type=float, default=1.0, metavar='F',
                    help='M6-T9: Feldanteil des Rückbaus mit --nur-rueckbau (Standard 1)')
    ap.add_argument('--rueckbau-zielfeld', type=float, default=None, metavar='F',
                    help='M6-T9: mit --nur-rueckbau Feldanteil am Ende (Standard wie '
                         '--rueckbau-feld); dazwischen Feldstufe mit den synth-Optionen')
    ap.add_argument('--osc-gewicht', type=float, default=None,
                    help='Gewicht von sine_condition im OSC-Versuch (0: ohne; Standard: '
                         'Koma-Äquivalent bei 10 %% Feld, osc_weight_for_field(0.1) ≈ 13)')
    ap.add_argument('--zyklus', default=None, metavar='STUFEN',
                    help='M6-T10: zyklischer Ablauf ab dem realen System --start (z. B. Z2 nach '
                         'dem Rückbau): Stufen aus feld, farbe, apertur (Komma-Liste); je Stufe '
                         'Rückkehr zu Geisterplatten, Stufe, Z2, Rückbau am realen System, Z2')
    ap.add_argument('--zyklus-wiederholen', type=int, default=2, metavar='N',
                    help='M6-T10: höchstens N Durchläufe der Stufenliste (Standard 2)')
    ap.add_argument('--zyklus-runden', type=int, default=8, metavar='N',
                    help='M6-T10: Runden von synth_loop in Feld- und Aperturstufe (Standard 8)')
    ap.add_argument('--geister-lagen', default='aperture,field', metavar='LISTE',
                    help='M6-T10: Lagen der Rückkehr zu Geisterplatten (mikroptim insert '
                         '--ghost-positions; Standard aperture,field)')
    ap.add_argument('--farb-spektrum-rang', action='store_true',
                    help='M6-T11: Operator B/Farbregel mit Partnerglas nach dem spektralen Rest '
                         'über das Zielspektrum (synth --split-spectrum)')
    ap.add_argument('--farb-startradien', action='store_true',
                    help='M6-T11: Operator B mit zusätzlichem Zweig „Startradien des Glieds“ '
                         '(synth --split-member-start)')
    ap.add_argument('--farb-chromatisch', type=float, default=None, metavar='S',
                    help='M6-T11: Farbbedarf aus den chromatischen Seidel-Termen, Anteil S am '
                         'größten RMS (synth --color-chromatic S)')
    ap.add_argument('--farb-modul', choices=['doublet', 'strong'], default=None,
                    help='M6-T11: Null-Kittglied bzw. starkes Kittglied (Wissensbasis) einfügen, '
                         'wenn die Farbe dominiert und keine Linse teilbar ist '
                         '(synth --color-module)')
    ap.add_argument('--farb-modul-immer', action='store_true',
                    help='M6-T11: Kittglied-Modul statt Teilen (synth --color-module-always)')
    ap.add_argument('--farb-anomal', action='store_true',
                    help='M6-T11: Modul mit mindestens einem anomalen Glas, |ΔP_g,F| >= 0,01 '
                         '(anomalous_pgf; synth --color-module-anomalous)')
    ap.add_argument('--farb-ziel', choices=['zwei', 'voll'], default='zwei',
                    help='M6-T11: Farbstufe über zwei Wellenlängen zum vollen Spektrum (Standard, '
                         'wie M6-T10) oder direkt zum vollen Spektrum')
    ap.add_argument('--rueckbau-punkt', type=float, default=None, metavar='Q',
                    help='M6-T11: Rückbau im Zyklus bei Q × Parameter der Stufe (NA, Feld bzw. '
                         'Spektrumbreite; z. B. 0,85) statt am erreichten Punkt, danach zurück '
                         'zum erreichten Punkt')
    args = ap.parse_args()
    if args.rueckbau_punkt is not None and not 0.0 < args.rueckbau_punkt < 1.0:
        ap.error('--rueckbau-punkt muss in (0, 1) liegen')
    if (args.farb_spektrum_rang or args.farb_startradien or args.farb_chromatisch or
            args.farb_modul):
        args.farbregel = True
    if args.zyklus:
        return cycle_run(args)
    if args.apertur_zweig:
        return aperture_branch(args)
    if args.osc_versuch:
        if args.osc_gewicht is None:
            args.osc_gewicht = osc_weight_for_field(0.1)
        if os.path.abspath(args.out) == os.path.join(ROOT, 'docs', 'ergebnisse', 'm6-t6'):
            args.out = os.path.join(ROOT, 'docs', 'ergebnisse', 'm11-t4')
        return osc_branch(args)
    FORCED['min_gap'] = args.erzwungen_min_spalt
    FORCED['stop_gap'] = args.erzwungen_blende
    extra = args.erzwungen_mehr
    r = Runner(os.path.abspath(args.exe), os.path.abspath(args.out), gpu=args.gpu,
               synth_options=synth_options(args))
    rings = ['--rings', '16']
    relax = ['--max-iterations', '1000', '-q'] + rings
    common = ['--curvature-bound', '1.2', '--max-steps', '3000', '--max-iterations', '500'] +         (['--element-budget', str(args.elementbudget)] if args.elementbudget else
         ['--max-elements', '24']) + rings
    summary = {'start': args.start, 'reference_sphere': args.referenzsphaere,
               'synth_options': r.synth_options, 'na_field': args.na_feld,
               'forced': dict(FORCED, extra=extra), 'stages': {}}

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
    set_reference_sphere(s, args.referenzsphaere)  # M6-T6b (ADR-0017)
    if args.farbe_von_anfang:
        # polychromatisch ab dem Start (Zhang: Farbe früh im Entwurf); die Gläser werden so nicht
        # erst monochromatisch gewählt (Lauf v9: F_B ≈ 0,0017 + 48·w² über der Spektrumbreite w,
        # also mehrere Wellen Farbe am Ende der monochromatischen Stufen)
        wl, w = SPECTRA[args.farbe_von_anfang]
        set_spectrum(s, wl, w)
    start = r.path('p0_start.json')
    save(s, start)
    na1 = repr(args.na_feld)
    if args.nur_rueckbau:
        # M6-T9: nur Rückbau am Punkt der Startdatei (volles Feld), Ergebnis als Ende der
        # Feldstufe; danach Z2
        na0 = psf_numerical_aperture(s)
        f0 = repr(args.rueckbau_feld)
        f1 = repr(args.rueckbau_zielfeld if args.rueckbau_zielfeld is not None
                  else args.rueckbau_feld)
        p2 = r.synth('p2_r0', start, ['--target-na', repr(na0), '--start-field', f0,
                                      '--target-field', f1] + common)
        res = load(r.path('p2_r0_log.json'))['result']
        r2 = [{'round': 0, 'synth': 'p2_r0', 'status': res['status'], 'point': res['point'],
               'elements': res['elements'], 'start_elements': res['start_elements'],
               'removals': res.get('removals', 0),
               'removal_attempts': res.get('removal_attempts', 0),
               'seconds': r.times.get('p2_r0')}]
        print(f"  p2_r0: {res['status']}, {res['start_elements']} -> {res['elements']} Elemente",
              flush=True)
        stage('p2', {'result': p2, 'rounds': r2, 'prune_only': True})
        args.bis = 'p2'
    elif args.gemeinsam:
        # M6-T8: Stufen 1 und 2 gemeinsam, (NA, Feld) vom Start (bzw. Endpunkt der vorigen Runde)
        # geradlinig (bzw. mit Exponent) nach (na_feld, 100 %); Runden und erzwungenes Einfügen
        # wie p1 und p2 zusammen
        joint = ['--joint-aperture-field']
        if args.gemeinsam_exponent:
            joint += ['--joint-field-exponent', repr(args.gemeinsam_exponent)]

        def opts12(point):
            f = 0.1 if point is None else point['field']
            return ['--target-na', na1, '--start-field', repr(f), '--target-field', '1'] + \
                joint + common

        p2, r2 = synth_loop(r, 'p2', start, opts12, relax,
                            lambda p: p['aperture'] + p['field'], args.na_feld + 1.0,
                            restart=full_field, max_rounds=24 + 4 * extra, forced=4 + 2 * extra)
        stage('p2', {'result': p2, 'rounds': r2, 'joint': joint})
    else:
        if args.ab_p2:
            # M6-T8: Start ist schon das Ergebnis der Stufe 1 (NA = Pupillenradius/f)
            p1 = start
            pt = {'aperture': psf_numerical_aperture(s), 'field': 0.1}
            stage('p1', {'result': p1, 'skipped': True, 'point': pt})
        else:
            p1, r1 = synth_loop(r, 'p1', start,
                                ['--target-na', na1, '--start-field', '0.1', '--target-field',
                                 '0.1'] + common, relax, lambda p: p['aperture'], args.na_feld,
                                restart=full_field, max_rounds=12 + 2 * extra, forced=2 + extra)
            stage('p1', {'result': p1, 'rounds': r1})
            pt = last_point(r, r1)

        # Stufe 2: Feld 10 % -> 100 % bei dieser NA
        def opts2(point):
            f = 0.1 if point is None else point['field']
            return ['--target-na', repr(pt['aperture']), '--start-field', repr(f),
                    '--target-field', '1'] + common

        p2, r2 = synth_loop(r, 'p2', p1, opts2, relax, lambda p: p['field'], 1.0,
                            restart=full_field, max_rounds=12 + 2 * extra, forced=2 + extra)
        stage('p2', {'result': p2, 'rounds': r2})
    pt = last_point(r, r2)
    field = pt['field']

    def z2_candidate():
        # Zwischenkandidat: Ende der Feldstufe (volles Feld, monochromatisch), ohne
        # Geisterplatten nachoptimiert und bewertet, am erreichten Feldanteil der Feldstufe
        # (M6-T6b; im Lauf M6-T6 war er 1)
        s = load(p2)
        set_field(s, [a * field for a in FIELD_ANGLES], FIELD_WEIGHTS)
        z2_in = r.path('z2_punkt.json')
        save(s, z2_in)
        z2 = polish(r, 'z2', z2_in, ['--max-iterations', '2000', '-q'] + rings, spectrum='mono')
        stage('z2', {'result': z2, 'eval': evaluate(r, 'z2', z2), 'aperture': pt['aperture'],
                     'field': field})
        if field < 1.0:
            # M6-T8: Z2 auch bei vollem Feld bewertet (Vergleichstabelle 0/70/100 %)
            s = load(z2)
            set_field(s, FIELD_ANGLES, FIELD_WEIGHTS)
            full = r.path('z2_vollfeld_in.json')
            save(s, full)
            summary['stages']['z2']['eval_full_field'] = evaluate(r, 'z2_vollfeld', full)
            stage('z2', summary['stages']['z2'])

    if args.bis == 'p2':
        z2_candidate()
        print('fertig (bis p2):', p2)
        return 0

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
    if args.farbe_frueh and not args.farbe_von_anfang:
        # Farbe vor der Aperturstufe (bei NA und Feld des Endes von Stufe 2): Bei NA 0,605 hat
        # das monochromatische System keinen Abstand zur Schwelle mehr (Lauf v8: F_B 0,0051 schon
        # bei Spektrumbreite 1/1024), bei NA 0,5 schon
        p3_in = color_stages(p2, last_point(r, r2)['aperture'])
    p3, r3 = synth_loop(r, 'p3', p3_in,
                        ['--target-na', '0.95', '--start-field', repr(field), '--target-field',
                         repr(field)] + common, relax, lambda p: p['aperture'], 0.95,
                        restart=full_field, max_rounds=16 + 2 * extra, forced=3 + extra,
                        element_budget=3)
    stage('p3', {'result': p3, 'rounds': r3})
    pt = last_point(r, r3)
    na = pt['aperture']
    cur = p3 if (args.farbe_frueh or args.farbe_von_anfang) else color_stages(p3, na)

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

    z2_candidate()
    print('fertig:', final)
    return 0


if __name__ == '__main__':
    sys.exit(main())
