"""Build and freeze the 10-case transfer sample (PR-MCI Phase 2 held-out partition).
Same evidence format and rules as case_studies/pr_mci_claude_derivation/build_manifests.py.
Labels are claim-level extensions written before any engine call on this sample."""
from __future__ import annotations

import csv, hashlib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pr_mci_claude_code"))
sys.path.insert(0, str(HERE.parent / "pr_mci_claude_derivation"))
import parse  # noqa: E402
from build_manifests import build_sources  # noqa: E402  (shared evidence format)

csv.field_size_limit(10**9)
PRMCI = Path(os.environ.get("PRMCI_DIR", HERE.parent / "pr_mci_claude_code" / "data" / "PR-MCI"))
ROUTING = {"derived": "PASS", "refuted": "RETRY", "insufficient_evidence": "ESCALATE"}

CASES = {
    "3119422525": dict(field="body", claim="add localization tests to verify keys, structure, and placeholders across translation files",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["apps/web/client/test/messages.test.ts añadido: tests 'same keys', 'same structure (value types)', 'valid interpolation placeholders' sobre los archivos de messages/"],
        reasoning="Enlaza tres tests añadidos (claves, estructura, placeholders) con los archivos de traducción que recorren."),
    "3242736120": dict(field="title", claim="Run elixir rosetta tests",
        situation="S3", provenance="replica", expected="insufficient_evidence", alternatives=[], inference=False,
        minimal_evidence=["único cambio: README de Elixir 'Compiled programs: 29/216' -> '28/239'"],
        reasoning="El diff no evidencia la ejecución de tests; solo un cambio de recuentos en un README."),
    "2855386414": dict(field="body", claim="This PR implements the merge and split functions for the PDF manipulation package.",
        situation="S2", provenance="replica", expected="refuted", alternatives=["insufficient_evidence"], inference=True,
        minimal_evidence=["src/index.ts: los cuerpos cambiados son remove e insert (stubs `return new ArrayBuffer(0)` eliminados)",
                          "merge/split solo aparecen en la línea de import de base ('-') y en contexto sin cambios"],
        reasoning="El PR implementa remove/insert; merge y split ya existían en base. Citar el import de base o el contexto como implementación es premisa inválida."),
    "3189027271": dict(field="title", claim="Support legacy arg patterns in tpc‑ds queries",
        situation="S2", provenance="replica", expected="refuted", alternatives=["insufficient_evidence"], inference=True,
        minimal_evidence=["runtime/vm/vm.go: solo cambian líneas de comentario", "`args = args[:fn.NumParams]` es contexto sin cambios (existía en base)"],
        reasoning="El PR no añade el soporte: el comportamiento ya existía en base y solo se reescribe el comentario. 'derived' apoyado en la línea de contexto sería falsa aceptación."),
    "3222826115": dict(field="body", claim="update COBOL machine translation checklist",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["tests/machine/x/cobol/README.md: '- [ ] slice.mochi' (base) -> '- [x] slice.mochi' (head)", "'(60/97 compiled)' -> '(61/97 compiled)'"],
        reasoning="Enlaza el cambio base->head del checklist con el archivo de traducciones COBOL."),
    "3242794608": dict(field="body", claim="remove its error file and add blank output file",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["call-a-function-8.error (removed)", "call-a-function-8.out (added) con una única línea vacía"],
        reasoning="Dos hechos: archivo de error eliminado y archivo de salida añadido vacío para el mismo programa."),
    "3148951832": dict(field="body", claim="include tests and notes on common Mochi errors",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["bloques `test \"...\"` añadidos", "comentario 'Common Mochi language errors and how to fix them'"],
        reasoning="Enlaza los bloques de test y el bloque de notas del mismo archivo añadido."),
    "3248046560": dict(field="body", claim="refine boolean printing when generating Rust code",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=["derived"], inference=False,
        minimal_evidence=["transpiler.go: emitArg imprime los argumentos bool como `if a { 1 } else { 0 }` salvo 'in'/'contains'"],
        reasoning="Se ve el nuevo camino de impresión; que sea una 'mejora' correcta exige ejecución. 'derived' aceptable solo citando emitArg y declarando el supuesto."),
    "3243568724": dict(field="body", claim="adjust docker build context",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=[".github/workflows/ci.yml: contexto de `docker build` '.' (base) -> 'alpha_factory_v1/demos/alpha_agi_insight_v1' (head)"],
        reasoning="Requiere vincular la línea '-' (base) y la '+' (head) del mismo comando."),
    "3222703182": dict(field="body", claim="restore machine README and list remaining tasks",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=["refuted"], inference=False,
        minimal_evidence=["tests/machine/x/c/README.md 'modified' con líneas de contexto existentes: existía en base", "+10 tareas sin marcar"],
        reasoning="'list remaining tasks' se sostiene; 'restore' (que faltara y se repusiera) no lo apoya la evidencia: el README existía en base."),
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main(force: bool = False) -> None:
    fin, fexp = HERE / "frozen/inputs_manifest.json", HERE / "frozen/expected_manifest.json"
    if (fin.exists() or fexp.exists()) and not force:
        sys.exit("Frozen manifests exist; refusing to overwrite.")
    sel = json.loads((HERE / "selection_log.json").read_text())
    val = {r["pr_id"]: r for r in csv.DictReader(open(PRMCI / "validation/validation_results.csv", encoding="utf-8-sig"))}
    prs = {p["pr_id"]: p for ag in ["OpenAI_Codex", "Copilot", "Devin", "Cursor"] for p in parse.load(ag, prmci_dir=str(PRMCI))}
    ids = sorted(s["pr_id"] for s in sel["selected"])
    assert set(ids) == set(CASES), set(ids) ^ set(CASES)
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    inputs, expected = [], []
    for i, pid in enumerate(ids, 1):
        cid, c, p = f"T{i:02d}", CASES[pid], prs[pid]
        assert norm(c["claim"]) in norm(p["title"] if c["field"] == "title" else p["body"]), (pid, c["claim"])
        meta = json.loads((HERE / "github_cache" / f"{pid}_pull.json").read_text())
        files = json.loads((HERE / "github_cache" / f"{pid}_files.json").read_text())
        s = next(x for x in sel["selected"] if x["pr_id"] == pid)
        inputs.append({"case_id": cid, "pr_id": pid, "agent": s["agent"], "url": p["url"], "base_sha": meta["base"]["sha"],
                       "head_sha": meta["head"]["sha"], "claim": c["claim"], "claim_source_field": c["field"],
                       "claim_author": f"{s['agent']} (PR author agent)", "net_patch_sha256": sha("\n".join(f.get("patch", "") for f in files)),
                       "sources": build_sources(meta, files)})
        v = val[pid]
        expected.append({"case_id": cid, "pr_id": pid, "situation": c["situation"], "provenance": c["provenance"],
                         "expected_verdict": c["expected"], "expected_routing": ROUTING[c["expected"]],
                         "acceptable_alternative_verdicts": c["alternatives"], "minimal_evidence": c["minimal_evidence"],
                         "acceptable_reasoning": c["reasoning"], "requires_inference": c["inference"],
                         "prmci_gold_pr_label": (v["consensus_label"] or v["annotator1_label"]).lower(),
                         "prmci_annotator_notes": {"a1": v["annotator1_notes"], "a2": v["annotator2_notes"], "consensus": v["consensus_notes"]}})
    (HERE / "frozen").mkdir(exist_ok=True)
    fin.write_text(json.dumps({"study": "PR-MCI Phase-2 transfer sample (held-out agents)", "prmci_commit": "ac6bbf4deab044d86389a6e178441292306bedcf", "cases": inputs}, indent=1, ensure_ascii=False) + "\n")
    fexp.write_text(json.dumps({"verdict_names": "claimlayer.v4 epistemic_verdict", "cases": expected}, indent=1, ensure_ascii=False) + "\n")
    h1, h2 = sha(fin.read_text()), sha(fexp.read_text())
    (HERE / "frozen/MANIFEST_HASHES.txt").write_text(f"{h1}  inputs_manifest.json\n{h2}  expected_manifest.json\n")
    print("inputs", h1); print("expected", h2)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
