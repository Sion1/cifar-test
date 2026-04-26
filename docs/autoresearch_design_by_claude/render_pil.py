#!/usr/bin/env python3
"""Render the 4 autoresearch design flowcharts as PNGs using pure PIL.

Each diagram is described as a grid of nodes with (col, row, w, h, text, kind).
Edges connect nodes by id; routing uses orthogonal segments through midpoints.
No external services, no system packages required.
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path("/radish/xl/zsl_project/Hysyn-ZSL-v3-SUN-autoresearch/figs/autoresearch_design_by_claude")
OUT.mkdir(parents=True, exist_ok=True)

# ---- font discovery ----
def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if pathlib.Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

FONT_TITLE = load_font(28)
FONT_NODE  = load_font(15)
FONT_EDGE  = load_font(12)
FONT_LEGEND = load_font(13)

# ---- color palette (kind → fill, stroke) ----
PALETTE = {
    "step":  ("#E3F2FD", "#1565C0"),
    "gate":  ("#F3E5F5", "#7B1FA2"),
    "warn":  ("#FFF4E6", "#E67E22"),
    "ok":    ("#E8F5E9", "#2E7D32"),
    "err":   ("#FFEBEE", "#C62828"),
    "decision": ("#FFFDE7", "#F9A825"),
    "found": ("#FFF9C4", "#F9A825"),
    "m1":    ("#BBDEFB", "#1565C0"),
    "m2":    ("#C8E6C9", "#2E7D32"),
    "m3":    ("#FFCCBC", "#D84315"),
    "m4":    ("#E1BEE7", "#6A1B9A"),
    "loss":  ("#FFE0B2", "#E65100"),
    "cycle": ("#E1F5FF", "#0277BD"),
    "neutral":("#FAFAFA","#424242"),
}

CELL_W = 280     # pixel width per grid column
CELL_H = 90      # pixel height per grid row
PAD_X  = 18      # horizontal pad between cells
PAD_Y  = 16      # vertical pad between cells
MARGIN = 60
TITLE_H = 70
LEGEND_H = 80

def wrap_text(text, font, max_width, draw):
    """Word-wrap by pixel width."""
    lines_out = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        cur = ""
        for w in words:
            trial = w if not cur else cur + " " + w
            bbox = draw.textbbox((0,0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                cur = trial
            else:
                if cur:
                    lines_out.append(cur)
                cur = w
        if cur:
            lines_out.append(cur)
    return lines_out

def draw_node(draw, x0, y0, x1, y1, text, kind, font):
    fill, stroke = PALETTE.get(kind, PALETTE["neutral"])
    radius = 10
    if kind == "decision":
        # diamond
        cx = (x0+x1)/2
        cy = (y0+y1)/2
        pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
        draw.polygon(pts, fill=fill, outline=stroke)
    elif kind in ("step","gate","warn","ok","err","cycle","found","m1","m2","m3","m4","loss","neutral"):
        draw.rounded_rectangle((x0,y0,x1,y1), radius=radius, fill=fill, outline=stroke, width=2)
    # text
    lines = wrap_text(text, font, (x1-x0)-12, draw)
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + 4
    total_h = len(lines) * line_h
    ty = (y0+y1)/2 - total_h/2
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        tw = bbox[2] - bbox[0]
        tx = (x0+x1)/2 - tw/2
        draw.text((tx, ty), ln, fill="#000", font=font)
        ty += line_h

def grid_to_px(col, row, span_c=1, span_r=1, x_off=MARGIN, y_off=MARGIN+TITLE_H):
    x0 = x_off + col*(CELL_W+PAD_X)
    y0 = y_off + row*(CELL_H+PAD_Y)
    x1 = x0 + span_c*CELL_W + (span_c-1)*PAD_X
    y1 = y0 + span_r*CELL_H + (span_r-1)*PAD_Y
    return x0,y0,x1,y1

def edge_endpoints(a_box, b_box):
    """Pick best port (top/bottom/left/right midpoints) on a and b for arrow."""
    ax0,ay0,ax1,ay1 = a_box
    bx0,by0,bx1,by1 = b_box
    acx, acy = (ax0+ax1)/2, (ay0+ay1)/2
    bcx, bcy = (bx0+bx1)/2, (by0+by1)/2
    # Decide based on relative position
    dx = bcx - acx
    dy = bcy - acy
    if abs(dy) >= abs(dx):
        # vertical-dominant
        if dy > 0:
            return (acx, ay1), (bcx, by0)  # bottom of a → top of b
        else:
            return (acx, ay0), (bcx, by1)
    else:
        if dx > 0:
            return (ax1, acy), (bx0, bcy)
        else:
            return (ax0, acy), (bx1, bcy)

def draw_arrow(draw, p0, p1, label=None):
    x0,y0 = p0
    x1,y1 = p1
    # orthogonal route: go to midpoint vertically/horizontally
    if abs(x1-x0) < 4 or abs(y1-y0) < 4:
        # straight
        pts = [p0, p1]
    else:
        if abs(y1-y0) >= abs(x1-x0):
            mid_y = (y0+y1)/2
            pts = [p0, (x0, mid_y), (x1, mid_y), p1]
        else:
            mid_x = (x0+x1)/2
            pts = [p0, (mid_x, y0), (mid_x, y1), p1]
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill="#444", width=2)
    # arrowhead
    import math
    ax,ay = pts[-2]
    bx,by = pts[-1]
    ang = math.atan2(by-ay, bx-ax)
    L = 10
    aw = 0.5
    p_left  = (bx - L*math.cos(ang-aw), by - L*math.sin(ang-aw))
    p_right = (bx - L*math.cos(ang+aw), by - L*math.sin(ang+aw))
    draw.polygon([p1, p_left, p_right], fill="#444")
    if label:
        # place label near midpoint of segment
        if len(pts) >= 4:
            mx = (pts[1][0] + pts[2][0]) / 2
            my = (pts[1][1] + pts[2][1]) / 2
        else:
            mx = (pts[0][0] + pts[-1][0]) / 2
            my = (pts[0][1] + pts[-1][1]) / 2
        bbox = draw.textbbox((0,0), label, font=FONT_EDGE)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        # white halo
        rect = (mx-tw/2-3, my-th/2-2, mx+tw/2+3, my+th/2+2)
        draw.rectangle(rect, fill="#FFFFFF", outline="#BBB")
        draw.text((mx-tw/2, my-th/2), label, fill="#1A237E", font=FONT_EDGE)

def render_diagram(filename, title, nodes, edges, n_cols, n_rows, legend_items=None):
    """nodes = {id: (col, row, span_c, span_r, text, kind)}; edges = [(a,b,label_or_None)]"""
    W = MARGIN*2 + n_cols*CELL_W + (n_cols-1)*PAD_X
    H = MARGIN*2 + TITLE_H + n_rows*CELL_H + (n_rows-1)*PAD_Y + (LEGEND_H if legend_items else 0)
    img = Image.new("RGB", (W,H), "white")
    draw = ImageDraw.Draw(img)
    # title
    draw.text((MARGIN, MARGIN-30), title, fill="#0D47A1", font=FONT_TITLE)
    # boxes
    boxes = {}
    for nid, spec in nodes.items():
        col, row, sc, sr, text, kind = spec
        x0,y0,x1,y1 = grid_to_px(col,row,sc,sr)
        boxes[nid] = (x0,y0,x1,y1,kind)
        draw_node(draw, x0,y0,x1,y1, text, kind, FONT_NODE)
    # edges
    for e in edges:
        a, b = e[0], e[1]
        label = e[2] if len(e) > 2 else None
        if a not in boxes or b not in boxes:
            continue
        ab = boxes[a][:4]
        bb = boxes[b][:4]
        p0, p1 = edge_endpoints(ab, bb)
        draw_arrow(draw, p0, p1, label)
    # legend
    if legend_items:
        ly = MARGIN+TITLE_H + n_rows*CELL_H + (n_rows-1)*PAD_Y + 30
        lx = MARGIN
        draw.text((lx, ly-22), "Legend:", fill="#000", font=FONT_LEGEND)
        for kind, label in legend_items:
            fill, stroke = PALETTE.get(kind, PALETTE["neutral"])
            draw.rounded_rectangle((lx, ly, lx+22, ly+18), radius=4, fill=fill, outline=stroke, width=2)
            draw.text((lx+30, ly+1), label, fill="#000", font=FONT_LEGEND)
            bbox = draw.textbbox((0,0), label, font=FONT_LEGEND)
            lx += 22 + 8 + (bbox[2]-bbox[0]) + 24
    img.save(OUT / filename)
    print(f"saved {OUT/filename}  ({W}x{H})")

# ============================================================================
# FIGURE 1 — Main loop scheduler
# ============================================================================
nodes1 = {
    # col, row, span_c, span_r, text, kind
    "start":   (1, 0, 3, 1, "External tmux wrapper\n(while true; do bash loop.sh; sleep 300; done)\nNOTE: 5-min cadence is external — loop.sh does not enforce it", "neutral"),
    "tick":    (2, 1, 1, 1, "loop.sh tick begins\nsource state/.consensus.env + .crossval.env", "step"),

    "g1":      (2, 2, 1, 1, "AUTORES_HOST_TAG\nexported?", "decision"),
    "x_refuse":(4, 2, 1, 1, "Refuse tick / exit 0", "err"),
    "g2":      (2, 3, 1, 1, "state/.loop.enabled.TAG\nsentinel exists?", "decision"),
    "g3":      (2, 4, 1, 1, "Acquire flock fd 9\non /tmp/autores.LOCK", "decision"),

    "s0":      (2, 5, 1, 1, "Step 0 · sanity\nclaude / program.md / state.tsv", "step"),
    "s1":      (2, 6, 1, 1, "Step 1 · Reap\nscan rows status=running", "step"),

    "r1":      (0, 7, 1, 1, "/proc/PID alive\nlocally?", "decision"),
    "rkeep":   (0, 8, 1, 1, "keep running", "ok"),
    "r2":      (1, 7, 1, 1, "tail+fstat exp log\n(flush CIFS mtime)", "step"),
    "r3":      (2, 7, 1, 1, "FINISH rc=N\nin log?", "decision"),
    "rcomp":   (3, 7, 1, 1, "rc=0 → completed", "ok"),
    "rfail":   (4, 7, 1, 2, "WARN: rc≠0 → FAILED\n=== DEAD-END STATE ===\nnot auto-analyzed\nnot git-committed\nNOT counted in STOP rule\nrequires manual fix", "warn"),
    "r4":      (2, 8, 1, 1, "mtime age > 900s?", "decision"),
    "rcomp2":  (3, 8, 1, 1, "yes → completed\n(heartbeat timeout)", "ok"),

    "s2":      (2, 9, 1, 1, "Step 2 · any row\nstatus=completed?", "decision"),
    "anlz":    (4, 9, 1, 1, "yes → see Figure 2\nAnalyze + async Consensus", "step"),
    "x_anlz":  (4,10, 1, 1, "exit 0\n(blocks propose this tick)", "ok"),

    "s3":      (2,10, 1, 1, "Step 3 · STOP criteria\nLAUNCHED≥80 (note: comment says 50)\nlast 5 verdict=Failure ≥ 3\nbest_h ≥ 45", "step"),
    "x_stop":  (0,10, 1, 1, "any hit → STOP exit 0", "err"),

    "s4":      (2,11, 1, 1, "Step 4 · running count\n≥ MAX_CONCURRENT?", "decision"),
    "x_wait":  (0,11, 1, 1, "yes → exit 0 / wait", "ok"),

    "gate":    (2,12, 1, 1, "Propose-Gate\n(standalone, not a Step)", "gate"),
    "gate_off":(0,12, 1, 1, "AUTORES_CONSENSUS_ENABLED\n≠ 1 → pass through", "ok"),
    "gate2":   (2,13, 1, 1, "find LATEST_ANALYZED:\nmax iter where status=analyzed\nAND verdict ≠ Bug", "step"),
    "gate3":   (2,14, 1, 1, "consensus.final.md\nexists for that iter?", "decision"),
    "x_gate":  (0,14, 1, 1, "no → exit 0\n(propose blocked)", "err"),
    "gate4":   (4,14, 1, 1, "WARN: gate ONLY checks file exists\ndoes NOT verify STATUS=\nCONSENSUS / OVERRIDE / PARSE_FAIL\nKNOWN DESIGN GAP", "warn"),

    "s5pre":   (2,15, 1, 1, "Step 5 · GPU pre-check\nnvidia-smi most-free GPU\nfree_mem ≥ 24 GB\nskip AUTORES_SKIP_GPUS", "step"),
    "x_no":    (4,15, 1, 1, "no GPU → exit 0", "ok"),

    "s5":      (2,16, 1, 1, "Step 5 · Propose\nWARN: FREE_GPU is PRE-CHECK ONLY\nNOT passed to run_experiment.sh\ntraining picks GPU independently", "warn"),
    "s5b":     (4,16, 1, 1, "if consensus enabled:\nbuild CONSENSUS_HINT (BINDING)\nfrom prev consensus.final.md\nWARN: PARSE_FAIL content also concatenated\nbut next-step usually fallbacks to §8", "warn"),

    "s5d":     (2,17, 1, 1, "timeout 900s claude -p PROPOSE\n--max-turns 60\nreads program.md / CLAUDE.md /\nALL iter logs / state.tsv", "step"),
    "s5e":     (2,18, 1, 1, "Write configs/ablation_shm/*.yaml\noptionally edit src/hysyn_zsl/*.py\ncall: bash run_experiment.sh CONFIG NEXT_ITER", "step"),
    "s5f":     (2,19, 1, 1, "run_experiment.sh:\nown GPU select + reservation +\nOOM preflight\nnohup python3 train.py\nappend state.tsv: status=running", "step"),
    "x_p":     (2,20, 1, 1, "exit 0", "ok"),
}
edges1 = [
    ("start","tick"), ("tick","g1"),
    ("g1","g2","yes"), ("g1","x_refuse","no"),
    ("g2","g3","yes"), ("g2","x_refuse","no"),
    ("g3","s0","got lock"), ("g3","x_refuse","busy"),
    ("s0","s1"), ("s1","r1"), ("s1","r2"), ("s1","r3"),
    ("r1","rkeep","yes"),
    ("r2","r3"),
    ("r3","rcomp","rc=0"), ("r3","rfail","rc≠0"), ("r3","r4","no FINISH"),
    ("r4","rcomp2","yes"), ("r4","rkeep","no"),
    ("rkeep","s2"), ("rcomp","s2"), ("rcomp2","s2"), ("rfail","s2"),
    ("s2","anlz","yes"), ("anlz","x_anlz"),
    ("s2","s3","no"),
    ("s3","x_stop","STOP hit"),
    ("s3","s4","none"),
    ("s4","x_wait","yes"),
    ("s4","gate","no"),
    ("gate","gate_off","disabled"),
    ("gate","gate2","enabled"),
    ("gate2","gate3"),
    ("gate3","x_gate","no"),
    ("gate3","gate4","yes"),
    ("gate4","s5pre"),
    ("gate_off","s5pre"),
    ("s5pre","x_no","no GPU"),
    ("s5pre","s5","FREE_GPU"),
    ("s5","s5b"),
    ("s5","s5d"),
    ("s5b","s5d"),
    ("s5d","s5e"),
    ("s5e","s5f"),
    ("s5f","x_p"),
]
legend = [("step","Step / action"), ("decision","Decision"), ("gate","Propose-Gate"),
          ("warn","Warning / known gap"), ("ok","OK / safe exit"), ("err","Refused / blocked")]

render_diagram("01_main_loop.png",
               "Figure 1 — loop.sh Main Scheduler State Machine (drawn by Claude · 2026-04-26)",
               nodes1, edges1, n_cols=5, n_rows=21, legend_items=legend)

# ============================================================================
# FIGURE 2 — Analyze + async consensus dispatch
# ============================================================================
nodes2 = {
    "a0":  (1, 0, 1, 1, "Step 2 fired:\nfirst row status=completed → ITER", "step"),
    "a1":  (1, 1, 1, 1, "timeout 1800s\nclaude -p ANALYZE iter NNN\n--max-turns 150", "step"),
    "a2":  (1, 2, 1, 2, "Claude executes 7 mandatory steps:\n1. Read program.md / CLAUDE.md / LAST 3 iter logs\n2. torch.load final.pth → metrics\n3. Write logs/iteration_NNN.md skeleton\n4. Update state.tsv: analyzed/best_h/verdict\n5. Generate figs/iter_NNN/ — 4 viz (per_class, gamma, t-SNE, attn)\n6. Refill §5 viz takeaways\n7. Append CLAUDE.md Documented findings", "step"),

    "a3":  (1, 4, 1, 1, "logs/iteration_NNN.md\nexists?", "decision"),
    "abug":(3, 4, 1, 2, "WARN: defensive guard fires when:\n• rc=0 silent failure  • rc=124 timeout  • any other failure\nMark row: analyzed / -1 / Bug\nSKIP git_iter_commit + consensus\n(prevents 80K-token-per-tick burn)", "warn"),
    "exit_bug": (3, 6, 1, 1, "exit 0", "ok"),

    "a4":  (1, 5, 1, 1, "bash scripts/git_iter_commit.sh ITER", "step"),
    "a4a": (1, 6, 1, 1, "git checkout -b autoresearch/iter-NNN", "step"),
    "a4b": (1, 7, 1, 1, "stage configs / logs / figs /\nstate.tsv / CLAUDE.md / scripts", "step"),
    "a4c": (1, 8, 1, 1, "structured commit message\n(Hypothesis / Results / Verdict)", "step"),
    "a4d": (1, 9, 1, 1, "AUTORES_GIT_AUTOPUSH\n=1 by DEFAULT (push enabled)\nnote: file's top comment says\n'default no' — code ≠ comment", "decision"),
    "a4e": (3, 9, 1, 1, "=1 → git push -u origin\nautoresearch/iter-NNN\n+ try gh pr create", "ok"),
    "a4f": (-1, 9, 1, 1, "=0 → local branch\nawaits user review", "ok"),

    "a4g": (1,10, 1, 2, "WARN: critical RESTORE step\nafter `git checkout PREV_BRANCH`,\nrun `git checkout BRANCH -- file`\nto restore working-tree copies of:\niteration_NNN.md / state.tsv / CLAUDE.md / figs/iter_NNN\nelse async consensus cannot find primary md\n→ infinite analyze loop (iter045 fix)", "warn"),

    "a5":  (1,12, 1, 1, "AUTORES_CONSENSUS_\nENABLED=1?", "decision"),
    "a6":  (1,13, 1, 2, "setsid nohup\nscripts/consensus_iter.sh ITER\n9>&- closes inherited flock fd\n(else consensus blocks all subsequent\nticks for ~13min — iter47 25-min stall fix)", "warn"),
    "a7":  (1,15, 1, 1, "loop.sh exits IMMEDIATELY\n(does NOT wait R1/R3/R2/R5)\nconsensus runs independently — see Figure 3", "ok"),
    "exit_main": (1,16, 1, 1, "exit 0", "ok"),
}
edges2 = [
    ("a0","a1"), ("a1","a2"), ("a2","a3"),
    ("a3","abug","no"), ("abug","exit_bug"),
    ("a3","a4","yes"),
    ("a4","a4a"), ("a4a","a4b"), ("a4b","a4c"), ("a4c","a4d"),
    ("a4d","a4e","=1 default"), ("a4d","a4f","=0"),
    ("a4e","a4g"), ("a4f","a4g"),
    ("a4g","a5"),
    ("a5","exit_main","no"),
    ("a5","a6","yes"),
    ("a6","a7"),
    ("a7","exit_main"),
]
# Adjust grid: a4f is in col -1, shift everything by +1 col
nodes2 = {k:( (v[0]+1,)+v[1:] ) for k,v in nodes2.items()}
render_diagram("02_analyze_subflow.png",
               "Figure 2 — Analyze Step + Async Consensus Dispatch (drawn by Claude · 2026-04-26)",
               nodes2, edges2, n_cols=5, n_rows=17, legend_items=legend)

# ============================================================================
# FIGURE 3 — 5-cycle consensus chain
# ============================================================================
nodes3 = {
    "c1":  (1, 0, 1, 1, "Cycle 1 done:\nmain analyze wrote iteration_NNN.md", "cycle"),
    "c2":  (1, 1, 1, 1, "Cycle 2 — eval R1\nSEQUENTIAL across:\nclaude / codex / gemini", "cycle"),
    "c2a": (1, 2, 1, 1, "each agent reads:\nprimary md + final.pth +\nper_class.csv + gamma_sweep.json", "step"),
    "c2b": (1, 3, 1, 1, "writes iteration_NNN.consensus.AGENT.r1.md\nverdict: AGREE / DISAGREE / COUNTER-PROPOSE", "step"),

    "c3":  (1, 4, 1, 1, "Cycle 3 — main revise", "cycle"),
    "c3a": (1, 5, 1, 1, "main claude reads 3 R1 critiques", "step"),
    "c3b": (1, 6, 1, 1, "0 DISAGREE?", "decision"),
    "c3c": (3, 6, 1, 1, "yes → HOLD\n(keep §8)", "ok"),
    "c3d": (-1, 6, 1, 1, "no → critique\ntechnically valid?", "decision"),
    "c3e": (-1, 7, 1, 1, "valid → REVISE\n(overwrite primary §8)", "ok"),
    "c3f": (1, 8, 1, 1, "write main.r3.md", "step"),

    "c4":  (1, 9, 1, 1, "Cycle 4 — eval R2\n3 evals re-judge final §8", "cycle"),
    "c4a": (1,10, 1, 1, "write AGENT.r2.md\n(AGREE / DISAGREE)", "step"),

    "c5":  (1,11, 1, 1, "Cycle 5 — main final", "cycle"),
    "c5a": (1,12, 1, 1, "R2 all AGREE?", "decision"),
    "c5b": (3,12, 1, 1, "yes → STATUS = CONSENSUS", "ok"),
    "c5c": (-1,12, 1, 1, "no → STATUS = OVERRIDE_BY_MAIN\nmain's §8 still wins\n(quality gate, not veto gate)", "warn"),
    "c5d": (1,13, 1, 1, "write main.r5.md", "step"),

    "cfin": (1,14, 1, 1, "python3 scripts/parse_consensus.py ITER", "step"),
    "cfina":(1,15, 1, 2, "3-LAYER FALLBACK for NEXT_STEP:\n(1) R5 '## Final next-step' section\n(2) primary §8\n(3) placeholder 'NEXT_STEP NOT EXTRACTED'", "warn"),
    "cfinb":(1,17, 1, 1, "write logs/iteration_NNN.consensus.final.md", "step"),
    "cstatus":(1,18, 1, 1, "STATUS field", "decision"),
    "cok":  (3,18, 1, 1, "CONSENSUS / OVERRIDE_BY_MAIN /\nPARSE_FAIL → file exists\n→ Propose-Gate releases\nWARN: gate does NOT verify STATUS", "warn"),
    "cerr": (-1,18, 1, 1, "file not written →\npropose permanently blocked\nneed manual stub OR\nAUTORES_CONSENSUS_ENABLED=0", "err"),
}
nodes3 = {k:( (v[0]+1,)+v[1:] ) for k,v in nodes3.items()}
edges3 = [
    ("c1","c2"),("c2","c2a"),("c2a","c2b"),("c2b","c3"),
    ("c3","c3a"),("c3a","c3b"),
    ("c3b","c3c","yes"),("c3b","c3d","no"),
    ("c3d","c3e","valid"),("c3d","c3c","not valid"),
    ("c3c","c3f"),("c3e","c3f"),
    ("c3f","c4"),("c4","c4a"),("c4a","c5"),
    ("c5","c5a"),
    ("c5a","c5b","yes"),("c5a","c5c","no"),
    ("c5b","c5d"),("c5c","c5d"),
    ("c5d","cfin"),("cfin","cfina"),("cfina","cfinb"),
    ("cfinb","cstatus"),
    ("cstatus","cok","file exists"),("cstatus","cerr","absent"),
]
legend3 = [("cycle","Consensus cycle"), ("step","Action"), ("decision","Decision"),
           ("warn","Warning / fallback"), ("ok","OK"), ("err","Permanent block")]
render_diagram("03_consensus_chain.png",
               "Figure 3 — 5-Cycle Consensus Chain (async background ~13min · drawn by Claude · 2026-04-26)",
               nodes3, edges3, n_cols=5, n_rows=20, legend_items=legend3)

# ============================================================================
# FIGURE 4 — Training data / module dependency flow (LR layout)
# ============================================================================
nodes4 = {
    "y":   (0, 0, 1, 1, "YAML config\nconfigs/ablation_shm/\nSUN_v2_*.yaml", "neutral"),
    "t":   (1, 0, 1, 1, "train.py", "step"),
    "tc":  (2, 0, 1, 1, "load_config /\nseed / wandb", "step"),
    "td":  (2, 1, 1, 1, "build_dataloader\nSUN Xian 645/72 split", "step"),
    "bb":  (2, 2, 1, 1, "DINO ViT-B/16 backbone\n(foundational, ALWAYS ON)", "found"),
    "ap":  (2, 3, 1, 1, "AttributePrototype\n(foundational, ALWAYS ON)\nlearnable matrix P", "found"),

    "gf":  (3, 2, 1, 1, "global feature", "step"),
    "lf":  (3, 4, 1, 1, "local patch features\n+ DINO CLS→patch attention", "step"),

    "ceg": (4, 2, 1, 1, "loss_ce_global\nCE on sim(global, P, class_attr)\nuses AttributePrototype\nweight λ_global\nALWAYS ON, never ablated", "loss"),

    "m1g": (4, 4, 1, 1, "M1 · Local Branch\nfeature_type=global+local\nis_attn=True", "m1"),
    "m1pool":(5,4, 1, 1, "if M1 on:\nper-attribute region pooling", "m1"),
    "noloc":(5, 5, 1, 1, "if M1 off:\nglobal-only path", "neutral"),

    "m2g": (5, 3, 1, 1, "M2 · DINO Prior\nREQUIRES M1 ON\nis_prior=True/False", "m2"),
    "m2h": (6, 2, 1, 1, "hard:\nthreshold mask prior", "m2"),
    "m2s": (6, 3, 1, 1, "soft_add:\nlog-bias prior\n(α or learnable_alpha)", "m2"),
    "m2n": (6, 4, 1, 1, "off:\nno prior", "m2"),

    "ap2": (7, 3, 1, 1, "local attribute prediction", "step"),
    "cel": (8, 3, 1, 1, "loss_ce_local\nweight λ_local", "loss"),

    "m4g": (7, 5, 1, 1, "M4 · VFA\nREQUIRES M1 ON\nis_feature_aggregation=True", "m4"),
    "m4a": (8, 5, 1, 1, "if M4 on:\nattribute-weighted\nfeature aggregation", "m4"),
    "cew": (8, 6, 1, 1, "loss_ce_wa\n(third CE pathway)", "loss"),

    "m3g": (4, 7, 1, 1, "M3 · Decorrelation\nINDEPENDENT of M1/M2/M4\nuse_ap_loss=True/False", "m3"),
    "m3l": (5, 7, 1, 1, "L_ap = ‖off-diag(PᵀP)‖_F\napplies to whole P\nweight w (ap_loss_weight)", "m3"),

    "sum": (8, 8, 1, 1, "total_loss =\nλ_g·L_g + λ_l·L_l +\nλ_w·L_wa + w·L_ap", "loss"),
    "opt": (9, 8, 1, 1, "optimizer step\nAdamW + cosine LR", "step"),
    "eval":(9, 5, 1, 1, "evaluate_v2 each epoch", "step"),
    "m":   (9, 4, 1, 1, "per-class CZSL/U/S/H\n+ γ-sweep + AUSUC", "step"),
    "bt":  (9, 3, 1, 1, "BestTracker", "step"),
    "ckpt":(9, 2, 1, 1, "best_h.pth /\nbest_czsl.pth /\nfinal.pth\n(contains ckpt['metrics'])", "ok"),
}
edges4 = [
    ("y","t"),("t","tc"),("t","td"),("t","bb"),("t","ap"),
    ("bb","gf"),("bb","lf"),
    ("ap","ceg"), ("gf","ceg"),
    ("lf","m1g"), ("m1g","m1pool"), ("m1g","noloc"),
    ("m1pool","m2g"),
    ("m2g","m2h","hard"), ("m2g","m2s","soft_add"), ("m2g","m2n","off"),
    ("m2h","ap2"),("m2s","ap2"),("m2n","ap2"),
    ("ap2","cel"),
    ("ap2","m4g"), ("m4g","m4a"), ("m4a","cew"),
    ("ap","m3g"), ("m3g","m3l"),
    ("ceg","sum"),("cel","sum"),("cew","sum"),("m3l","sum"),
    ("sum","opt"),("opt","eval"),("eval","m"),("m","bt"),("bt","ckpt"),
]
legend4 = [("found","Foundational (always on)"),("m1","M1 Local Branch"),("m2","M2 DINO Prior (req M1)"),
           ("m3","M3 Decorrelation (indep)"),("m4","M4 VFA (req M1)"),("loss","Loss term"),("step","Pipeline step")]
# Make it wider
CELL_W_BACKUP = CELL_W
render_diagram("04_training_modules.png",
               "Figure 4 — Training Data / Module Dependency Flow (drawn by Claude · 2026-04-26)",
               nodes4, edges4, n_cols=10, n_rows=10, legend_items=legend4)
