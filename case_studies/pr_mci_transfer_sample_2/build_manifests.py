"""Build and freeze the second 10-case transfer sample (M5; PR-MCI Phase 2 held-out partition, excluding
the first transfer sample). Same evidence format as the first sample.

Labels were written from the PR text and the net diff BEFORE any engine call on this sample and
BEFORE the stage-3 planner prompt was changed.

Claim rule (declared before labelling): the claim is the PR title verbatim, unless the title is not a
statement of a change made by the PR (a [WIP] marker plus an issue restatement); then it is the first
body bullet that states a change made by the PR. Verdict scale: derived / refuted /
insufficient_evidence (claims of fixes, behaviour or execution are insufficient from a textual diff
unless the change itself is textual, e.g. documentation)."""
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
    "3159544492": dict(field="title", claim="Expand COBOL backend tests",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["tests/compiler/cobol/if_else.{mochi,cob.out,out} y while_loop.{mochi,cob.out,out} añadidos"],
        reasoning="Los archivos de test añadidos en tests/compiler/cobol amplían los tests del backend COBOL (vincular directorio y archivos añadidos)."),
    "3077280318": dict(field="title", claim="Fix snippet deletion",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=["derived"], inference=False,
        minimal_evidence=["snippetmanager.py: delete() reordena `idaapi.netnode(...).kill()` antes de `snippet.netnode_idx = BADNODE`"],
        reasoning="El cambio textual es visible; que corrija el borrado es una afirmación de comportamiento. 'derived' solo con razonamiento semántico explícito sobre el orden (antes se mataba BADNODE)."),
    "3206231303": dict(field="title", claim="Remove integrate-boards option from generate-script command",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["GenerateScriptCommand.cs: '-AddOption(IntegrateBoards);'", "GenerateScriptCommand.cs: definición `--integrate-boards` eliminada"],
        reasoning="Vincula la eliminación del registro de la opción y de su definición en el comando generate-script."),
    "3104613438": dict(field="body", claim="Created test case that demonstrates the problem",
        situation="S1", provenance="extension", expected="derived", alternatives=["insufficient_evidence"], inference=True,
        minimal_evidence=["LiteralSurfaceTest.scala añadido con test(\"reproduce the exact issue from the problem statement\")"],
        reasoning="Título [WIP] que reformula la incidencia: se usa la primera viñeta de cambio. El test añadido existe; 'demuestra' el problema requeriría ejecutarlo, por eso se admite insufficient_evidence."),
    "3147343923": dict(field="title", claim="Add LeetCode 190 solution in Mochi",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["examples/leetcode/190/reverse-bits.mochi añadido: '// Solution for LeetCode problem 190 - Reverse Bits', fun reverseBits"],
        reasoning="Vincula el número 190 (ruta y comentario) con el archivo .mochi añadido que define la solución."),
    "3104056039": dict(field="title", claim="Fix TreeView auto-expansion when opening trace files",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=["derived"], inference=False,
        minimal_evidence=["MainWindow.xaml: '+<Setter Property=\"IsExpanded\" Value=\"{Binding IsExpanded}\" />'"],
        reasoning="Se ve el binding añadido; que arregle la autoexpansión al abrir trazas es comportamiento en ejecución."),
    "3244968620": dict(field="title", claim="Enhance Fortran transpiler golden tests",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["typed_let.{f90,out} y var_assignment.{f90,out} añadidos en tests/transpiler/x/fortran", "transpiler_test.go: flag -update y helpers añadidos"],
        reasoning="Vincula archivos golden añadidos y el arnés de test modificado del transpilador Fortran."),
    "3250736652": dict(field="title", claim="[alpha_factory] add diagnostics to server tests",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["tests/test_api_server_subprocess.py: '+raise AssertionError(f\"server exited with ...{output}\")' y lectura de proc.stdout añadidas"],
        reasoning="Las aserciones añadidas que incluyen la salida del servidor son diagnósticos en los tests del servidor."),
    "3245817025": dict(field="title", claim="Fix README numbering and server reference",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=True,
        minimal_evidence=["README.md: '-5. Compile and upload' -> '+4. Compile and upload'", "README.md: 'overview of the client code' -> 'overview of the server code'"],
        reasoning="Arreglo textual de documentación: dos pares '-'/'+' en README.md. Control de sobreabstención: 'Fix' aquí no es comportamiento."),
    "3088525099": dict(field="title", claim="[alpha_factory] Add quick links to AGENTS",
        situation="S1", provenance="extension", expected="derived", alternatives=[], inference=False,
        minimal_evidence=["AGENTS.md: bloque '+**Quick links**' con cuatro enlaces a secciones"],
        reasoning="Un único bloque añadido en AGENTS.md."),
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
    fin.write_text(json.dumps({"study": "PR-MCI Phase-2 second transfer sample (held-out agents, M5 reserve)", "prmci_commit": "ac6bbf4deab044d86389a6e178441292306bedcf", "cases": inputs}, indent=1, ensure_ascii=False) + "\n")
    fexp.write_text(json.dumps({"verdict_names": "claimlayer.v4 epistemic_verdict", "cases": expected}, indent=1, ensure_ascii=False) + "\n")
    h1, h2 = sha(fin.read_text()), sha(fexp.read_text())
    (HERE / "frozen/MANIFEST_HASHES.txt").write_text(f"{h1}  inputs_manifest.json\n{h2}  expected_manifest.json\n")
    print("inputs", h1); print("expected", h2)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
