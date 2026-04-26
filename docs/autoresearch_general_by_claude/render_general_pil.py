#!/usr/bin/env python3
"""Render the 4 GENERAL autoresearch design flowcharts as PNGs (no task-specific terms)."""
import pathlib, math
from PIL import Image, ImageDraw, ImageFont

OUT = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

def load_font(size):
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if pathlib.Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

FONT_TITLE  = load_font(28)
FONT_NODE   = load_font(15)
FONT_EDGE   = load_font(12)
FONT_LEGEND = load_font(13)

PALETTE = {
    "step":  ("#E3F2FD", "#1565C0"),
    "gate":  ("#F3E5F5", "#7B1FA2"),
    "warn":  ("#FFF4E6", "#E67E22"),
    "ok":    ("#E8F5E9", "#2E7D32"),
    "err":   ("#FFEBEE", "#C62828"),
    "decision": ("#FFFDE7", "#F9A825"),
    "found": ("#FFF9C4", "#F9A825"),
    "opt":   ("#BBDEFB", "#1565C0"),
    "optreq":("#C8E6C9", "#2E7D32"),
    "ind":   ("#FFCCBC", "#D84315"),
    "obj":   ("#FFE0B2", "#E65100"),
    "cycle": ("#E1F5FF", "#0277BD"),
    "neutral":("#FAFAFA","#424242"),
}

CELL_W = 290
CELL_H = 92
PAD_X  = 18
PAD_Y  = 16
MARGIN = 60
TITLE_H = 70
LEGEND_H = 80

def wrap_text(text, font, max_width, draw):
    out = []
    for line in text.split("\n"):
        words = line.split(" ")
        cur = ""
        for w in words:
            trial = w if not cur else cur + " " + w
            bbox = draw.textbbox((0,0), trial, font=font)
            if bbox[2]-bbox[0] <= max_width:
                cur = trial
            else:
                if cur: out.append(cur)
                cur = w
        if cur: out.append(cur)
    return out

def draw_node(draw, x0,y0,x1,y1, text, kind, font):
    fill, stroke = PALETTE.get(kind, PALETTE["neutral"])
    if kind == "decision":
        cx,cy = (x0+x1)/2, (y0+y1)/2
        draw.polygon([(cx,y0),(x1,cy),(cx,y1),(x0,cy)], fill=fill, outline=stroke)
    else:
        draw.rounded_rectangle((x0,y0,x1,y1), radius=10, fill=fill, outline=stroke, width=2)
    lines = wrap_text(text, font, (x1-x0)-12, draw)
    line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + 4
    ty = (y0+y1)/2 - len(lines)*line_h/2
    for ln in lines:
        bbox = draw.textbbox((0,0), ln, font=font)
        tx = (x0+x1)/2 - (bbox[2]-bbox[0])/2
        draw.text((tx,ty), ln, fill="#000", font=font)
        ty += line_h

def grid_to_px(col,row,sc=1,sr=1, x_off=MARGIN, y_off=MARGIN+TITLE_H):
    x0 = x_off + col*(CELL_W+PAD_X)
    y0 = y_off + row*(CELL_H+PAD_Y)
    x1 = x0 + sc*CELL_W + (sc-1)*PAD_X
    y1 = y0 + sr*CELL_H + (sr-1)*PAD_Y
    return x0,y0,x1,y1

def edge_endpoints(a,b):
    ax0,ay0,ax1,ay1 = a
    bx0,by0,bx1,by1 = b
    acx,acy = (ax0+ax1)/2,(ay0+ay1)/2
    bcx,bcy = (bx0+bx1)/2,(by0+by1)/2
    dx,dy = bcx-acx, bcy-acy
    if abs(dy) >= abs(dx):
        return ((acx,ay1),(bcx,by0)) if dy>0 else ((acx,ay0),(bcx,by1))
    else:
        return ((ax1,acy),(bx0,bcy)) if dx>0 else ((ax0,acy),(bx1,bcy))

def draw_arrow(draw, p0, p1, label=None):
    x0,y0 = p0; x1,y1 = p1
    if abs(x1-x0)<4 or abs(y1-y0)<4:
        pts = [p0,p1]
    elif abs(y1-y0) >= abs(x1-x0):
        my = (y0+y1)/2
        pts = [p0,(x0,my),(x1,my),p1]
    else:
        mx = (x0+x1)/2
        pts = [p0,(mx,y0),(mx,y1),p1]
    for i in range(len(pts)-1):
        draw.line([pts[i],pts[i+1]], fill="#444", width=2)
    ax,ay = pts[-2]; bx,by = pts[-1]
    ang = math.atan2(by-ay, bx-ax); L=10; aw=0.5
    pl = (bx-L*math.cos(ang-aw), by-L*math.sin(ang-aw))
    pr = (bx-L*math.cos(ang+aw), by-L*math.sin(ang+aw))
    draw.polygon([p1,pl,pr], fill="#444")
    if label:
        if len(pts)>=4:
            mx = (pts[1][0]+pts[2][0])/2; my = (pts[1][1]+pts[2][1])/2
        else:
            mx = (pts[0][0]+pts[-1][0])/2; my = (pts[0][1]+pts[-1][1])/2
        bbox = draw.textbbox((0,0), label, font=FONT_EDGE)
        tw,th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.rectangle((mx-tw/2-3,my-th/2-2,mx+tw/2+3,my+th/2+2), fill="#FFF", outline="#BBB")
        draw.text((mx-tw/2,my-th/2), label, fill="#1A237E", font=FONT_EDGE)

def render(filename, title, nodes, edges, n_cols, n_rows, legend_items=None):
    W = MARGIN*2 + n_cols*CELL_W + (n_cols-1)*PAD_X
    H = MARGIN*2 + TITLE_H + n_rows*CELL_H + (n_rows-1)*PAD_Y + (LEGEND_H if legend_items else 0)
    img = Image.new("RGB",(W,H),"white")
    d = ImageDraw.Draw(img)
    d.text((MARGIN, MARGIN-30), title, fill="#0D47A1", font=FONT_TITLE)
    boxes = {}
    for nid, spec in nodes.items():
        col,row,sc,sr,text,kind = spec
        x0,y0,x1,y1 = grid_to_px(col,row,sc,sr)
        boxes[nid] = (x0,y0,x1,y1)
        draw_node(d, x0,y0,x1,y1, text, kind, FONT_NODE)
    for e in edges:
        a,b = e[0],e[1]; lab = e[2] if len(e)>2 else None
        if a not in boxes or b not in boxes: continue
        p0,p1 = edge_endpoints(boxes[a], boxes[b])
        draw_arrow(d, p0, p1, lab)
    if legend_items:
        ly = MARGIN+TITLE_H + n_rows*CELL_H + (n_rows-1)*PAD_Y + 30
        lx = MARGIN
        d.text((lx,ly-22), "Legend:", fill="#000", font=FONT_LEGEND)
        for kind,lbl in legend_items:
            fill,stroke = PALETTE.get(kind, PALETTE["neutral"])
            d.rounded_rectangle((lx,ly,lx+22,ly+18), radius=4, fill=fill, outline=stroke, width=2)
            d.text((lx+30,ly+1), lbl, fill="#000", font=FONT_LEGEND)
            bbox = d.textbbox((0,0), lbl, font=FONT_LEGEND)
            lx += 22+8+(bbox[2]-bbox[0])+24
    img.save(OUT/filename)
    print(f"saved {OUT/filename}  ({W}x{H})")

# ============================================================================
# FIGURE 1 — Scheduler (general)
# ============================================================================
n1 = {
    "start":   (1, 0, 3, 1, "External cadence wrapper\n(cron / tmux / systemd timer)\nNOTE: cadence is set externally, not by the loop", "neutral"),
    "tick":    (2, 1, 1, 1, "Driver tick begins\nsource per-tick env overrides", "step"),
    "g1":      (2, 2, 1, 1, "Host identity tag set?", "decision"),
    "x_refuse":(4, 2, 1, 1, "Refuse tick / exit 0\n(prevents silent multi-host\nraces on shared FS)", "err"),
    "g2":      (2, 3, 1, 1, "Per-host enable sentinel\nfile exists?", "decision"),
    "g3":      (2, 4, 1, 1, "Acquire single-tick lock\n(local FS, NOT shared FS)", "decision"),
    "s0":      (2, 5, 1, 1, "Step 0 · sanity\nLLM CLI / rules file /\nstate ledger", "step"),
    "s1":      (2, 6, 1, 1, "Step 1 · Reap\nscan ledger for status=running", "step"),
    "r1":      (0, 7, 1, 1, "local PID alive?", "decision"),
    "rkeep":   (0, 8, 1, 1, "keep status=running", "ok"),
    "r2":      (1, 7, 1, 1, "Read experiment log tail\nforce FS-cache flush via fstat\n(shared-FS staleness fix)", "step"),
    "r3":      (2, 7, 1, 1, "FINISH signal\nin log?", "decision"),
    "rcomp":   (3, 7, 1, 1, "exit code 0\n→ COMPLETED", "ok"),
    "rfail":   (4, 7, 1, 2, "WARN: exit code != 0\n→ FAILED\n=== DEAD-END STATE ===\nnot auto-analyzed\nnot auto-committed\nNOT counted in STOP\nrequires manual fix", "warn"),
    "r4":      (2, 8, 1, 1, "log mtime stale\nbeyond heartbeat threshold?", "decision"),
    "rcomp2":  (3, 8, 1, 1, "yes → COMPLETED\n(heartbeat timeout)", "ok"),
    "s2":      (2, 9, 1, 1, "Step 2 · any COMPLETED\nawaiting analysis?", "decision"),
    "anlz":    (4, 9, 1, 1, "yes → see Figure 2\nanalysis + async consensus", "step"),
    "x_anlz":  (4,10, 1, 1, "exit 0\n(blocks proposals\nthis tick)", "ok"),
    "s3":      (2,10, 1, 1, "Step 3 · STOP heuristics\n• iteration budget reached\n• failure rate too high\n• target metric reached", "step"),
    "x_stop":  (0,10, 1, 1, "any hit → STOP exit 0", "err"),
    "s4":      (2,11, 1, 1, "Step 4 · concurrent-experiment\ncap reached?", "decision"),
    "x_wait":  (0,11, 1, 1, "yes → exit 0 / wait", "ok"),
    "gate":    (2,12, 1, 1, "Propose-Gate\n(optional; standalone block)", "gate"),
    "gate_off":(0,12, 1, 1, "consensus disabled\n→ pass through", "ok"),
    "gate2":   (2,13, 1, 1, "find latest analyzed iter\n(skip bug-marked rows)", "step"),
    "gate3":   (2,14, 1, 1, "consensus output\nfile present?", "decision"),
    "x_gate":  (0,14, 1, 1, "no → exit 0\n(consensus still running)", "err"),
    "gate4":   (2,15, 1, 1, "consensus STATUS\nfield valid?", "decision"),
    "x_gate2": (0,15, 1, 1, "PARSE_FAIL → exit 0\nrequires manual recovery\n(fix output OR\ndisable consensus)", "err"),
    "s5pre":   (2,16, 1, 1, "Step 5 · resource pre-check\nselect compute device\nwith sufficient capacity", "step"),
    "x_no":    (4,16, 1, 1, "no resource → exit 0", "ok"),
    "s5":      (2,17, 1, 1, "Step 5 · Proposal phase\nWARN: pre-checked resource is HINT\nlauncher re-selects at launch time", "warn"),
    "s5b":     (4,17, 1, 1, "if consensus enabled:\nbuild propose-context\nfrom consensus output\n(authority calibrated by STATUS:\nBINDING / STRONG / WEAK)", "warn"),
    "s5d":     (2,18, 1, 1, "Bounded LLM call · PROPOSE\nreads: rules / working memory /\nfull iter history / state ledger", "step"),
    "s5e":     (2,19, 1, 1, "Materialize spec:\nwrite config artifact +\noptionally edit codebase +\ninvoke experiment launcher", "step"),
    "s5f":     (2,20, 1, 1, "Launcher (separate process):\nown resource selection +\ncapacity preflight +\nbackground process spawn +\nappend ledger row status=running", "step"),
    "x_p":     (2,21, 1, 1, "exit 0", "ok"),
}
e1 = [
    ("start","tick"),("tick","g1"),
    ("g1","g2","yes"),("g1","x_refuse","no"),
    ("g2","g3","yes"),("g2","x_refuse","no"),
    ("g3","s0","got lock"),("g3","x_refuse","busy"),
    ("s0","s1"),("s1","r1"),("s1","r2"),("s1","r3"),
    ("r1","rkeep","yes"),
    ("r2","r3"),
    ("r3","rcomp","rc=0"),("r3","rfail","rc≠0"),("r3","r4","no FINISH"),
    ("r4","rcomp2","yes"),("r4","rkeep","no"),
    ("rkeep","s2"),("rcomp","s2"),("rcomp2","s2"),("rfail","s2"),
    ("s2","anlz","yes"),("anlz","x_anlz"),
    ("s2","s3","no"),
    ("s3","x_stop","STOP hit"),("s3","s4","none"),
    ("s4","x_wait","yes"),("s4","gate","no"),
    ("gate","gate_off","disabled"),("gate","gate2","enabled"),
    ("gate2","gate3"),
    ("gate3","x_gate","no"),
    ("gate3","gate4","yes"),
    ("gate4","x_gate2","PARSE_FAIL"),
    ("gate4","s5pre","CONSENSUS / OVERRIDE"),
    ("gate_off","s5pre"),
    ("s5pre","x_no","no resource"),("s5pre","s5","ok"),
    ("s5","s5b"),("s5","s5d"),("s5b","s5d"),
    ("s5d","s5e"),("s5e","s5f"),("s5f","x_p"),
]
legend = [("step","Step / action"),("decision","Decision"),("gate","Propose-Gate"),
          ("warn","Warning / known gap"),("ok","OK / safe exit"),("err","Refused / blocked")]
render("01_scheduler_general.png",
       "Figure 1 — Autonomous Research Loop · Scheduler State Machine (general · drawn by Claude)",
       n1, e1, n_cols=5, n_rows=22, legend_items=legend)

# ============================================================================
# FIGURE 2 — Analyze + async dispatch (general)
# ============================================================================
n2 = {
    "a0":  (1, 0, 1, 1, "Step 2 fired:\nfirst COMPLETED row → ITER", "step"),
    "a1":  (1, 1, 1, 1, "Bounded LLM call · ANALYZE\n(timeout + max-turns guards)", "step"),
    "a2":  (1, 2, 1, 2, "Analysis agent · skeleton-first workflow:\n1. Read rules / working memory / recent reports\n2. Load experiment results\n3. Write iter report SKELETON\n4. Update state ledger (analyzed/score/verdict)\n5. Generate mandatory artifacts (viz/tables)\n6. Refill skeleton with evidence\n7. Append findings to working memory", "step"),
    "a3":  (1, 4, 1, 1, "report file exists?", "decision"),
    "abug":(3, 4, 1, 2, "WARN: defensive guard fires when:\n• clean exit but no report (silent fail)\n• timeout (signal kill)\n• any other failure mode\nMark row: analyzed / NULL / Bug\nSKIP commit + consensus\n(prevents per-tick token burn loop)", "warn"),
    "exit_bug": (3, 6, 1, 1, "exit 0", "ok"),
    "a4":  (1, 5, 1, 1, "Invoke version-control\ncommit helper for ITER", "step"),
    "a4a": (1, 6, 1, 1, "Switch to per-iter branch\n(off main, never auto-merged)", "step"),
    "a4b": (1, 7, 1, 1, "Stage iter artifacts:\nconfig + report + viz +\nstate ledger + working memory", "step"),
    "a4c": (1, 8, 1, 1, "Build structured commit message\n(hypothesis / results / verdict)", "step"),
    "a4d": (1, 9, 1, 1, "auto-push toggle?\n(default: enabled)", "decision"),
    "a4e": (3, 9, 1, 1, "enabled → push branch\n+ open PR for human review", "ok"),
    "a4f": (-1, 9, 1, 1, "disabled → local branch\n(awaits manual push)", "ok"),
    "a4g": (1,10, 1, 2, "WARN: critical RESTORE step\nafter switching back to base branch,\ncherry-restore iter outputs into working tree:\nreport / state ledger / working memory\nWHY: downstream async consensus reads them\nFROM working tree, not from per-iter branch.\nSkipping → 'primary not found' →\ninfinite analyze loop", "warn"),
    "a5":  (1,12, 1, 1, "consensus mode enabled?", "decision"),
    "a6":  (1,13, 1, 2, "Detached background spawn\nof consensus pipeline:\n'setsid + nohup + close inherited fd' pattern\nWHY close inherited fd: parent holds\nthe per-tick lock fd; without explicit close,\nchild inherits → blocks ALL future ticks\nfor the full consensus runtime", "warn"),
    "a7":  (1,15, 1, 1, "Driver exits IMMEDIATELY\n(does NOT wait on consensus)\nsee Figure 3 for the chain", "ok"),
    "exit_main": (1,16, 1, 1, "exit 0", "ok"),
}
n2 = {k:((v[0]+1,)+v[1:]) for k,v in n2.items()}
e2 = [
    ("a0","a1"),("a1","a2"),("a2","a3"),
    ("a3","abug","no"),("abug","exit_bug"),
    ("a3","a4","yes"),
    ("a4","a4a"),("a4a","a4b"),("a4b","a4c"),("a4c","a4d"),
    ("a4d","a4e","enabled"),("a4d","a4f","disabled"),
    ("a4e","a4g"),("a4f","a4g"),
    ("a4g","a5"),
    ("a5","exit_main","no"),("a5","a6","yes"),
    ("a6","a7"),("a7","exit_main"),
]
render("02_analyze_general.png",
       "Figure 2 — Primary Analysis + Async Consensus Dispatch (general · drawn by Claude)",
       n2, e2, n_cols=5, n_rows=17, legend_items=legend)

# ============================================================================
# FIGURE 3 — Consensus chain (general)
# ============================================================================
n3 = {
    "c1":  (1, 0, 1, 1, "Cycle 1 (already done):\nprimary author wrote analysis\n(verdict + next-step proposal)", "cycle"),
    "c2":  (1, 1, 1, 1, "Cycle 2 — peer review round 1\nK independent reviewers\n(diverse model families for diversity)\nSEQUENTIAL execution", "cycle"),
    "c2a": (1, 2, 1, 1, "Each reviewer reads:\nprimary analysis + raw outputs +\nresearch rules", "step"),
    "c2b": (1, 3, 1, 1, "Each writes critique:\nAGREE/DISAGREE on verdict;\nAGREE/DISAGREE/COUNTER on next-step", "step"),
    "c3":  (1, 4, 1, 1, "Cycle 3 — primary revision", "cycle"),
    "c3a": (1, 5, 1, 1, "Primary author reads K critiques", "step"),
    "c3b": (1, 6, 1, 1, "any reviewer disagrees on next-step?", "decision"),
    "c3c": (3, 6, 1, 1, "no → HOLD\n(keep original next-step)", "ok"),
    "c3d": (-1, 6, 1, 1, "yes → critique\ntechnically valid?", "decision"),
    "c3e": (-1, 7, 1, 1, "valid → REVISE\n(rewrite next-step section)", "ok"),
    "c3f": (1, 8, 1, 1, "Write revision note\n(HOLD or REVISE)", "step"),
    "c4":  (1, 9, 1, 1, "Cycle 4 — peer review round 2\nK reviewers re-judge final next-step\n(short verdict only)", "cycle"),
    "c4a": (1,10, 1, 1, "Each writes:\nAGREE or DISAGREE", "step"),
    "c5":  (1,11, 1, 1, "Cycle 5 — primary final call", "cycle"),
    "c5a": (1,12, 1, 1, "round 2: all AGREE?", "decision"),
    "c5b": (3,12, 1, 1, "yes → STATUS = CONSENSUS", "ok"),
    "c5c": (-1,12, 1, 1, "no → STATUS = OVERRIDE_BY_PRIMARY\n(after N cycles without agreement,\nprimary's call wins per policy)\nQuality gate, NOT veto gate", "warn"),
    "c5d": (1,13, 1, 1, "Write final-call note", "step"),
    "cfin": (1,14, 1, 1, "Run aggregator script", "step"),
    "cfina":(1,15, 1, 2, "Aggregator extracts NEXT_STEP\nvia 3-LAYER FALLBACK:\n(1) primary's final-call section\n(2) primary's original next-step\n(3) placeholder 'NOT EXTRACTED'\nSTATUS: CONSENSUS / OVERRIDE / PARSE_FAIL", "warn"),
    "cfinb":(1,17, 1, 1, "Write canonical consensus output\n(propose phase reads this)", "step"),
    "cstatus":(1,18, 1, 1, "STATUS field", "decision"),
    "cok":  (3,18, 1, 1, "CONSENSUS → release gate\nHINT marked BINDING", "ok"),
    "cok2": (3,19, 1, 1, "OVERRIDE → release gate\nHINT marked STRONG\n(dissent noted)", "warn"),
    "cerr": (-1,18, 1, 1, "PARSE_FAIL → BLOCK\n(per aggregator's contract)\nrequires user fix\nor disable consensus", "err"),
    "cerr2":(-1,19, 1, 1, "file absent → BLOCK\nindefinitely until written", "err"),
}
n3 = {k:((v[0]+1,)+v[1:]) for k,v in n3.items()}
e3 = [
    ("c1","c2"),("c2","c2a"),("c2a","c2b"),("c2b","c3"),
    ("c3","c3a"),("c3a","c3b"),
    ("c3b","c3c","no"),("c3b","c3d","yes"),
    ("c3d","c3e","valid"),("c3d","c3c","not valid"),
    ("c3c","c3f"),("c3e","c3f"),
    ("c3f","c4"),("c4","c4a"),("c4a","c5"),
    ("c5","c5a"),
    ("c5a","c5b","yes"),("c5a","c5c","no"),
    ("c5b","c5d"),("c5c","c5d"),
    ("c5d","cfin"),("cfin","cfina"),("cfina","cfinb"),
    ("cfinb","cstatus"),
    ("cstatus","cok","CONSENSUS"),
    ("cstatus","cok2","OVERRIDE"),
    ("cstatus","cerr","PARSE_FAIL"),
    ("cstatus","cerr2","absent"),
]
legend3 = [("cycle","Consensus cycle"),("step","Action"),("decision","Decision"),
           ("warn","Warning / fallback"),("ok","OK / release"),("err","Permanent block")]
render("03_consensus_general.png",
       "Figure 3 — N-Cycle Multi-Agent Consensus (general · drawn by Claude)",
       n3, e3, n_cols=5, n_rows=21, legend_items=legend3)

# ============================================================================
# FIGURE 4 — Experiment execution (general, LR layout)
# ============================================================================
n4 = {
    "cfg":   (0, 0, 1, 1, "Experiment spec\n(YAML / JSON / TOML)", "neutral"),
    "lc":    (1, 0, 1, 1, "Experiment launcher\n(thin shell wrapper)", "step"),
    "pre":   (2, 0, 1, 1, "Pre-flight checks:\nresource select + capacity +\noutput-path policy", "step"),
    "init":  (2, 1, 1, 1, "Pipeline init:\nseed / logging / metrics", "step"),
    "data":  (2, 2, 1, 1, "Data layer\n(load / split / batch)", "step"),
    "core":  (3, 0, 1, 1, "Core component A\n(always-on backbone)", "found"),
    "coreB": (3, 1, 1, 1, "Core component B\n(always-on head)", "found"),
    "feat":  (4, 0, 1, 1, "Primary representation", "step"),
    "repr":  (4, 1, 1, 1, "Representation refinement", "step"),
    "objM":  (5, 0, 1, 1, "Main objective term\n(always present)", "obj"),

    "optA":  (4, 3, 1, 1, "Optional module A\n(ablation switch)", "opt"),
    "optAon":(5, 2, 1, 1, "if A on: active", "opt"),
    "optAoff":(5, 3, 1, 1, "if A off: skip", "neutral"),
    "objA":  (6, 2, 1, 1, "Auxiliary objective A", "obj"),

    "optB":  (5, 4, 1, 1, "Optional module B\n(REQUIRES A)", "optreq"),
    "optBon":(6, 4, 1, 1, "if B on: active", "optreq"),
    "objB":  (7, 4, 1, 1, "Auxiliary objective B", "obj"),

    "optC":  (5, 5, 1, 1, "Optional module C\n(REQUIRES A)", "optreq"),
    "optCon":(6, 5, 1, 1, "if C on: active", "optreq"),
    "objC":  (7, 5, 1, 1, "Auxiliary objective C", "obj"),

    "optD":  (4, 6, 1, 1, "Optional module D\n(INDEPENDENT)", "ind"),
    "optDon":(5, 6, 1, 1, "if D on: active", "ind"),
    "objD":  (6, 6, 1, 1, "Auxiliary objective D", "obj"),

    "sum":   (7, 1, 1, 1, "Composite objective\n= weighted sum of\nactive terms", "obj"),
    "step":  (8, 1, 1, 1, "Optimization step", "step"),
    "eval":  (8, 2, 1, 1, "Periodic evaluation\n(metrics + sweeps)", "step"),
    "track": (8, 3, 1, 1, "Best-tracker\n(running optimum)", "step"),
    "save":  (8, 4, 1, 1, "Persist checkpoints:\nbest / final / metrics", "step"),
    "fin":   (8, 5, 1, 1, "Write FINISH-line to log\n(authoritative end signal\nread by reaper in Figure 1)", "ok"),
}
e4 = [
    ("cfg","lc"),("lc","pre"),("pre","init"),("init","data"),
    ("init","core"),("init","coreB"),
    ("core","feat"),("coreB","repr"),
    ("feat","objM"),("repr","objM"),
    ("feat","optA"),
    ("optA","optAon","on"),("optA","optAoff","off"),
    ("optAon","objA"),
    ("optAon","optB"),("optB","optBon","on"),("optBon","objB"),
    ("optAon","optC"),("optC","optCon","on"),("optCon","objC"),
    ("repr","optD"),("optD","optDon","on"),("optDon","objD"),
    ("objM","sum"),("objA","sum"),("objB","sum"),("objC","sum"),("objD","sum"),
    ("sum","step"),("step","eval"),("eval","track"),("track","save"),("save","fin"),
]
legend4 = [("found","Foundational (always on)"),("opt","Optional module (ablation)"),
           ("optreq","Optional · requires another module"),("ind","Optional · independent"),
           ("obj","Objective term"),("step","Pipeline step")]
render("04_experiment_general.png",
       "Figure 4 — Experiment Execution Sub-Process (general · drawn by Claude)",
       n4, e4, n_cols=9, n_rows=8, legend_items=legend4)
