#!/usr/bin/env bash
# Mikroptim: neueste Version holen, bauen, testen, kurze Messläufe (Linux, z. B. herkules3).
#
# Aufruf:
#   tools/aktualisieren_und_testen.sh [Optionen]
#   MIKROPTIM_RECHNER=cpu tools/aktualisieren_und_testen.sh
#
# Rechnerprofil (Variable MIKROPTIM_RECHNER oder --rechner; Standard: herkules3):
#   herkules3  1 TB RAM, NVIDIA RTX 6000 Ada (48 GB, sm_89)    -> CUDA-Architektur 89-real, GPU 0
#   rtx3090    RTX 3090 (24 GB, sm_86) unter Linux              -> CUDA-Architektur 86-real, GPU 0
#   cpu        ohne GPU (Preset release-cpu)
# (Die Tesla P100 auf herkules ist seit ADR-0015 keine Zielhardware mehr.)
# Einzelwerte lassen sich überschreiben: MIKROPTIM_GPU (GPU-Nummer für --device, Standard aus dem
# Profil), MIKROPTIM_CUDA_ARCH (z. B. "60-real;89-real"), MIKROPTIM_PRESET.
# Gebaut wird nur für die Architektur des Profils (schneller als die volle Liste aus CMake). Die
# gewählte GPU wird über CUDA_VISIBLE_DEVICES für Tests und Messläufe sichtbar gemacht.
#
# Optionen:
#   --rechner NAME   Rechnerprofil (siehe oben)
#   --repo DIR       Klon von Mikroptim (Standard: $HOME/Mikroptim; fehlt er, wird geklont)
#   --branch NAME    Branch (Standard: main)
#   --preset NAME    CMake-Preset (Standard aus dem Profil)
#   --jobs N         parallele Bauprozesse (Standard: Zahl der Kerne)
#   --device N       GPU für die Messläufe (synth --gpu --device N; Standard aus dem Profil)
#   --arch LISTE     CUDA-Architekturen (Standard aus dem Profil)
#   --no-build       nicht bauen (vorhandenen Build verwenden)
#   --no-tests       ctest überspringen
#   --no-bench       Messläufe überspringen
#   --python         zusätzlich Python-Paket in einer venv bauen und pytest ausführen
#   --m6t6           zusätzlich den langen Lauf der Zielspezifikation mit aktuellen Standards und
#                    Farbregel (tools/m6t6_zielspezifikation.py --aktuell --farbregel)
#   --m6t6-dokumentiert  ebenso, aber wie im dokumentierten Lauf (synth --stall 0, ohne Farbregel)
#   --force          lokale Änderungen im Klon verwerfen (git reset --hard; sonst Abbruch)
#   --no-pull        Klon nicht aktualisieren (vorhandenen Stand testen)
#
# Ergebnis: Protokolle und Zusammenfassung unter $REPO/ergebnisse_lokal/<Zeitstempel>/.
# Exit-Code: 0 alles grün, 1 Tests oder Messläufe fehlgeschlagen, 2 Aufruf-/Git-/Baufehler.

set -uo pipefail

REPO="${HOME}/Mikroptim"
BRANCH="main"
RECHNER="${MIKROPTIM_RECHNER:-herkules3}"
PRESET="${MIKROPTIM_PRESET:-}"
CUDA_ARCH="${MIKROPTIM_CUDA_ARCH:-}"
JOBS="$(nproc 2>/dev/null || echo 4)"
DEVICE="${MIKROPTIM_GPU:-}"
DO_BUILD=1
DO_TESTS=1
DO_BENCH=1
DO_PYTHON=0
DO_M6T6=0
M6T6_ARGS=(--aktuell --farbregel)
FORCE=0
DO_PULL=1
REMOTE_URL="https://github.com/PhotonPower/Mikroptim.git"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --rechner) RECHNER="$2"; shift 2 ;;
        --arch) CUDA_ARCH="$2"; shift 2 ;;
        --no-build) DO_BUILD=0; shift ;;
        --no-tests) DO_TESTS=0; shift ;;
        --no-bench) DO_BENCH=0; shift ;;
        --python) DO_PYTHON=1; shift ;;
        --m6t6) DO_M6T6=1; shift ;;
        --m6t6-dokumentiert) DO_M6T6=1; M6T6_ARGS=(); shift ;;
        --force) FORCE=1; shift ;;
        --no-pull) DO_PULL=0; shift ;;
        -h|--help) sed -n '2,41p' "$0"; exit 0 ;;
        *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
    esac
done

# Rechnerprofil
case "$RECHNER" in
    herkules3) PROFIL_ARCH="89-real"; PROFIL_GPU="0"; PROFIL_PRESET="release-cuda"
               PROFIL_TEXT="herkules3: RTX 6000 Ada (48 GB, sm_89)" ;;
    rtx3090)   PROFIL_ARCH="86-real"; PROFIL_GPU="0"; PROFIL_PRESET="release-cuda"
               PROFIL_TEXT="RTX 3090 (24 GB, sm_86)" ;;
    herkules)  echo "Profil herkules (Tesla P100) entfällt seit ADR-0015" >&2; exit 2 ;;
    cpu)       PROFIL_ARCH="";        PROFIL_GPU="";  PROFIL_PRESET="release-cpu"
               PROFIL_TEXT="nur CPU" ;;
    *) echo "Unbekanntes Rechnerprofil '$RECHNER' (herkules3, rtx3090, cpu)" >&2; exit 2 ;;
esac
[[ -z "$PRESET" ]] && PRESET="$PROFIL_PRESET"
[[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="$PROFIL_ARCH"
[[ -z "$DEVICE" ]] && DEVICE="${PROFIL_GPU:-0}"
if [[ "$PRESET" == *cuda* ]] && ! command -v nvcc >/dev/null 2>&1; then
    echo "Profil $RECHNER verlangt CUDA, aber nvcc ist nicht im PATH (CUDA 12 oder 13 laden oder --rechner cpu)" >&2
    exit 2
fi
GPU=0
[[ "$PRESET" == *cuda* ]] && GPU=1
if [[ $GPU -eq 1 ]] && ! nvidia-smi -L >/dev/null 2>&1; then
    echo "Profil $RECHNER verlangt eine GPU, aber nvidia-smi erreicht keinen Treiber (Treiber prüfen oder --rechner cpu)" >&2
    exit 2
fi
# Tests und Messläufe sehen nur die GPU des Profils (dort Gerät 0)
[[ $GPU -eq 1 ]] && export CUDA_VISIBLE_DEVICES="$DEVICE"
echo "Rechnerprofil: $PROFIL_TEXT (Preset $PRESET${CUDA_ARCH:+, CUDA-Architektur $CUDA_ARCH}, GPU $DEVICE)"

FAILED=()
fail() { FAILED+=("$1"); echo "!! FEHLGESCHLAGEN: $1"; }
step() { echo; echo "==== $* ===="; }

# ---- 1. Quellen holen ------------------------------------------------------------------------
step "Quellen ($REPO, Branch $BRANCH)"
if [[ ! -d "$REPO/.git" ]]; then
    echo "Kein Klon gefunden, klone $REMOTE_URL"
    git clone "$REMOTE_URL" "$REPO" || { echo "git clone fehlgeschlagen" >&2; exit 2; }
fi
cd "$REPO" || exit 2
if [[ -n "$(git status --porcelain --untracked-files=no)" && $FORCE -eq 0 ]]; then
    echo "Lokale Änderungen im Klon – bitte sichern oder mit --force fortfahren:" >&2
    git status --short --untracked-files=no >&2
    exit 2
fi
OLD_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo -)"
if [[ $DO_PULL -eq 1 ]]; then
    git fetch --prune origin || { echo "git fetch fehlgeschlagen" >&2; exit 2; }
    git checkout "$BRANCH" || exit 2
    if [[ $FORCE -eq 1 ]]; then
        git reset --hard "origin/$BRANCH" || exit 2
    else
        git pull --ff-only origin "$BRANCH" || { echo "git pull --ff-only fehlgeschlagen" >&2; exit 2; }
    fi
fi
NEW_COMMIT="$(git rev-parse --short HEAD)"
echo "Stand: $OLD_COMMIT -> $NEW_COMMIT"
git log --oneline -8

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$REPO/ergebnisse_lokal/$STAMP"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/protokoll.txt") 2>&1
echo "Protokoll: $OUT/protokoll.txt"

# ---- 2. Umgebung -----------------------------------------------------------------------------
step "Umgebung"
{
    echo "Rechner: $(hostname)  Profil: $RECHNER  Datum: $(date -Iseconds)  Commit: $NEW_COMMIT  Preset: $PRESET"
    [[ $GPU -eq 1 ]] && echo "CUDA-Architektur: $CUDA_ARCH  GPU für Messläufe: $DEVICE"
    cmake --version | head -1
    (c++ --version || g++ --version) 2>/dev/null | head -1
    if [[ $GPU -eq 1 ]]; then
        nvcc --version | tail -2
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used --format=csv
    fi
} | tee "$OUT/umgebung.txt"

# ---- 3. Bauen --------------------------------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
    step "Konfigurieren und bauen ($PRESET, -j $JOBS)"
    CMAKE_EXTRA=()
    [[ $GPU -eq 1 && -n "$CUDA_ARCH" ]] && CMAKE_EXTRA+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
    cmake --preset "$PRESET" "${CMAKE_EXTRA[@]}" > "$OUT/konfigurieren.log" 2>&1 \
        || { tail -30 "$OUT/konfigurieren.log"; echo "Konfigurieren fehlgeschlagen" >&2; exit 2; }
    START=$(date +%s)
    cmake --build --preset "$PRESET" --parallel "$JOBS" > "$OUT/bauen.log" 2>&1 \
        || { grep -E "error|Error" "$OUT/bauen.log" | head -30; echo "Bauen fehlgeschlagen" >&2; exit 2; }
    echo "gebaut in $(( $(date +%s) - START )) s"
fi
EXE="$(find "build/$PRESET" -maxdepth 3 -type f -name mikroptim -perm -u+x | head -1)"
TESTS_EXE="$(find "build/$PRESET" -maxdepth 4 -type f -name mikroptim_tests -perm -u+x | head -1)"
if [[ -z "$EXE" ]]; then echo "mikroptim nicht gefunden unter build/$PRESET" >&2; exit 2; fi
echo "Programm: $EXE"

# ---- 4. Tests --------------------------------------------------------------------------------
if [[ $DO_TESTS -eq 1 ]]; then
    step "ctest ($PRESET)"
    START=$(date +%s)
    if ctest --preset "$PRESET" --output-on-failure --timeout 1800 > "$OUT/ctest.log" 2>&1; then
        echo "ctest grün"
    else
        fail "ctest"
        grep -E "Failed|\*\*\*|tests failed" "$OUT/ctest.log" | head -40
    fi
    tail -5 "$OUT/ctest.log"
    echo "ctest in $(( $(date +%s) - START )) s"

    if [[ $GPU -eq 1 && -n "$TESTS_EXE" ]]; then
        step "GPU: Mehrere Geräte und Prüfsummen (hardware.md, M7-T3)"
        ctest --preset "$PRESET" -R "MultiTraceEngine\." --output-on-failure > "$OUT/multi_gpu.log" 2>&1 \
            || fail "MultiTraceEngine"
        tail -3 "$OUT/multi_gpu.log"
        "$TESTS_EXE" --gtest_filter=MultiTraceEngine.ResultFingerprint 2>&1 \
            | grep -E "Gerät|RESIDUALS|MERIT|PREFILTER|OK|FAILED" | tee "$OUT/fingerprint.txt"
    fi
fi

# ---- 5. Python (optional) --------------------------------------------------------------------
if [[ $DO_PYTHON -eq 1 ]]; then
    step "Python-Paket und pytest"
    VENV="$REPO/.venv-test"
    python3 -m venv "$VENV" && source "$VENV/bin/activate"
    pip install -q --upgrade pip
    if pip install -q ".[glass]" pytest jsonschema > "$OUT/python_build.log" 2>&1; then
        (cd python/tests && python -m pytest -q) > "$OUT/pytest.log" 2>&1 || fail "pytest"
        tail -3 "$OUT/pytest.log"
    else
        fail "Python-Paket bauen"; tail -20 "$OUT/python_build.log"
    fi
    deactivate || true
fi

# ---- 6. Messläufe ----------------------------------------------------------------------------
# Läufe aus den letzten Änderungen (M11-T1c Stagnation, M11-T3 Prädiktor, M11-T5c Farbregel),
# auf der CPU und – falls vorhanden – auf der GPU.
if [[ $DO_BENCH -eq 1 ]]; then
    step "Messläufe"
    BENCH="$OUT/laeufe"
    mkdir -p "$BENCH"
    run() { # Name, Argumente …
        local name="$1"; shift
        local start
        start=$(date +%s%N)
        if "$EXE" "$@" --json --quiet -o "$BENCH/$name.json" > "$BENCH/${name}_log.json" 2> "$BENCH/${name}_err.txt"; then
            awk -v n="$name" -v a="$start" -v b="$(date +%s%N)" 'BEGIN { printf "%s %.1f\n", n, (b - a) / 1e9 }' >> "$BENCH/zeiten.txt"
        else
            fail "Lauf $name"; head -5 "$BENCH/${name}_err.txt"
        fi
    }
    H=data/systems/einzellinse_homotopie_start.json
    A=data/systems/einzellinse_achromat_start.json
    run "homotopie_standard"     synth "$H" --target-na 0.6
    run "homotopie_stall0"       synth "$H" --target-na 0.6 --stall 0
    run "homotopie_praediktor"   synth "$H" --target-na 0.6 --predictor
    run "achromat_ohne_regel"    synth "$A" --max-elements 3
    run "achromat_farbregel"     synth "$A" --max-elements 3 --color-rule
    if [[ $GPU -eq 1 ]]; then
        # CUDA_VISIBLE_DEVICES ist auf die GPU des Profils gesetzt: dort ist sie Gerät 0
        run "homotopie_gpu"         synth "$H" --target-na 0.6 --gpu --device 0
        run "achromat_farbregel_gpu" synth "$A" --max-elements 3 --color-rule --gpu --device 0
    fi
    "$EXE" aberrations data/systems/testsystem_mehrlinser.json > "$BENCH/aberrationen.txt" 2>&1 || fail "aberrations"
    "$EXE" glasses data/glass/synthetisch.json data/glass/basis.json --top 5 > "$BENCH/glaeser.txt" 2>&1 || fail "glasses"

    python3 - "$BENCH" << 'PY' | tee "$OUT/messlaeufe.txt"
import collections, json, os, sys
d = sys.argv[1]
zeiten = {}
if os.path.exists(os.path.join(d, "zeiten.txt")):
    for line in open(os.path.join(d, "zeiten.txt")):
        n, t = line.split()
        zeiten[n] = float(t)
print(f"{'Lauf':26} {'Status':16} {'Elem.':>5} {'RMS max':>9} {'Lösungen':>9} {'LM-Zyklen':>10} {'Zeit [s]':>9}  LM-Status")
for n in sorted(f[:-9] for f in os.listdir(d) if f.endswith("_log.json")):
    try:
        r = json.load(open(os.path.join(d, n + "_log.json")))["result"]
    except Exception as e:
        print(f"{n:26} (keine Ausgabe: {e})")
        continue
    st = r.get("steps", [])
    cnt = collections.Counter(s["lm"]["status"] for s in st if s.get("lm"))
    cyc = sum(s["lm"]["iterations"] for s in st if s.get("lm"))
    g = r.get("goal") or {}
    rms = g.get("max_rms_waves", g.get("max_rms"))
    rms_s = f"{rms:.4f}" if isinstance(rms, (int, float)) else "–"
    print(f"{n:26} {r['status']:16} {r['elements']:>5} {rms_s:>9} {r['solves']:>9} {cyc:>10} "
          f"{zeiten.get(n, float('nan')):>9.1f}  {dict(cnt)}")
PY
fi

# ---- 7. Zielspezifikation (optional, lang) ---------------------------------------------------
if [[ $DO_M6T6 -eq 1 ]]; then
    step "Zielspezifikation M6-T6 (lang)"
    ARGS=(--exe "$EXE" --out "$OUT/m6t6" "${M6T6_ARGS[@]}")
    [[ $GPU -eq 1 ]] && ARGS+=(--gpu)
    START=$(date +%s)
    python3 tools/m6t6_zielspezifikation.py "${ARGS[@]}" > "$OUT/m6t6.log" 2>&1 || fail "M6-T6"
    tail -20 "$OUT/m6t6.log"
    echo "M6-T6 in $(( ($(date +%s) - START) / 60 )) min"
fi

# ---- Zusammenfassung -------------------------------------------------------------------------
step "Zusammenfassung"
{
    echo "Commit $NEW_COMMIT (vorher $OLD_COMMIT), Profil $RECHNER, Preset $PRESET, $(hostname), $(date -Iseconds)"
    if [[ ${#FAILED[@]} -eq 0 ]]; then
        echo "ALLES GRÜN"
    else
        echo "FEHLGESCHLAGEN: ${FAILED[*]}"
    fi
    echo "Ergebnisse: $OUT"
} | tee "$OUT/zusammenfassung.txt"
[[ ${#FAILED[@]} -eq 0 ]] && exit 0 || exit 1
