"""Step 2: build and freeze the 20-case input and expected-answer manifests.

Inputs manifest  = exactly what the system receives: the claim (verbatim from the PR title or
                   body written by Claude Code) and admissible evidence (net base->head GitHub
                   diff of the PR, one neutral document per file, plus a diff summary).
Expected manifest = reference verdicts, minimal evidence and acceptable reasoning. It is never
                   sent to the planner. Case ids are neutral (C01..C20, ordered by pr_id).
Run once; refuses to overwrite frozen manifests unless --force is given.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pr_mci_claude_code"))
import parse  # noqa: E402

csv.field_size_limit(10**9)
PRMCI = Path(os.environ.get("PRMCI_DIR", HERE.parent / "pr_mci_claude_code" / "data" / "PR-MCI"))
FROZEN = HERE / "frozen"

# ---------------------------------------------------------------------------
# Case definitions (claim text is verbatim; expected answers written before any engine call).
# situation: S1 = holds but needs >=2 linked facts; S2 = plausible, evidence has a concrete
# counterexample; S3 = evidence supports a weaker claim but leaves the exact claim open.
# provenance: "replica" = the claim is the one PR-MCI annotators singled out for the PR label;
#             "extension" = claim chosen here from the PR text (claim-level label is new).
# ---------------------------------------------------------------------------
CASES = {
    "3206405036": dict(field="body", claim="Added comprehensive test for undefined opts scenario",
        situation="S3", provenance="replica", expected="insufficient_evidence", alternatives=["refuted"],
        minimal_evidence=["test/main.spec.ts: test 'flamethrower initializes without error when opts is undefined' (añadido en head)",
                          "el cuerpo del test define `const opts = undefined` y evalúa la expresión copiada dentro de un <script> inyectado; no importa ni invoca la función exportada de lib/main.ts"],
        reasoning="Se sostiene la afirmación débil 'se añadió un test con ese nombre para el escenario'. 'Comprehensive test' del escenario real no se demuestra: el test simula la expresión y no ejercita lib/main.ts. Refutada es aceptable si se cita expresamente que el test no llama al código bajo prueba. 'derived' sería falsa aceptación."),
    "3069923749": dict(field="body", claim="Adds fields for `cargo-insta` version and the output of `cargo insta --version` to the bug report issue template.",
        situation="S1", provenance="extension", expected="derived", alternatives=[],
        minimal_evidence=["+ id: cargo-insta-version / label: cargo-insta Version", "+ id: cargo-insta-command-version / label: cargo insta --version output",
                          "archivo .github/ISSUE_TEMPLATE/1-bug.yml (plantilla de bug)"],
        reasoning="Enlaza dos inputs añadidos (líneas +) con el archivo de plantilla de bug. La eliminación de `insta-version` (líneas −, estado base) no es premisa de la conclusión; el 'partial' de PR-MCI se debe a esa omisión, no a la falsedad de esta afirmación."),
    "3204772289": dict(field="body", claim="Added GUI Controller functionality to LangConnect for managing MCP (Model Context Protocol) servers through a web UI.",
        situation="S2", provenance="replica", expected="refuted", alternatives=["insufficient_evidence"],
        minimal_evidence=["resumen del diff: Files changed: 0, Lines added: 0, Lines deleted: 0"],
        reasoning="Un PR sin archivos ni líneas cambiadas no puede añadir funcionalidad: el diff vacío es el contraejemplo. Requiere la regla 'añadir código implica ≥1 archivo cambiado'. 'insufficient_evidence' es aceptable si se justifica por ausencia total de cambios; 'derived' sería falsa aceptación."),
    "3132739442": dict(field="title", claim="Consolidate algorithmic constants into comprehensive constants header",
        situation="S3", provenance="replica", expected="insufficient_evidence", alternatives=["refuted"],
        minimal_evidence=["añadido vmec_constants/vmec_algorithm_constants.h con constantes documentadas",
                          "radial_profiles.cc solo cambia un comentario; ningún punto de uso incluye ni usa el nuevo header",
                          "el propio header dice 'Also defined in vmec.h as kSignOfJacobian' / 'as kPDamp'"],
        reasoning="Se sostiene 'se añadió un header con constantes'. 'Consolidar' (trasladar a un único lugar) no se demuestra y hay indicios en contra (las constantes siguen definidas en vmec.h y ningún call site cambia). Refutada es aceptable si cita esos indicios. 'derived' sería falsa aceptación."),
    "2975165913": dict(field="body", claim="Updated all links pointing to /notebooks/Commands/ to now point to /docs/examples/Commands/ following repository reorganization.",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=[],
        minimal_evidence=["en los 6 archivos del diff, cada línea − con /notebooks/Commands/ tiene su línea + con /docs/examples/Commands/",
                          "el resto del repositorio (archivos no cambiados) no forma parte de la evidencia"],
        reasoning="La versión débil (los enlaces de los archivos del diff se actualizaron) se sostiene; 'all links' exige conocer archivos no cambiados. NOTA: la etiqueta PR-MCI 'misaligned' se basó en 12 archivos de código presentes en la representación del corpus pero ausentes del diff neto base→head; no es transferible a esta evidencia."),
    "3146561196": dict(field="body", claim="Build will fail if versions are inconsistent or package-lock.json is out of date",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=[],
        minimal_evidence=["+ step 'Check version consistency' con `run: npm run check-version` en .github/workflows/main.yml",
                          "scripts/check-version-consistency.js termina con process.exit(1) ante inconsistencias",
                          "FALTA: la definición del script npm `check-version` (package.json no está en el diff neto)"],
        reasoning="Cadena plausible con un eslabón ausente: nada en la evidencia enlaza `npm run check-version` con el script nuevo. 'derived' sería falsa aceptación. NOTA: la etiqueta 'partial' de PR-MCI cita Dockerfile/.dockerignore/GHCR, ausentes del diff neto."),
    "2889875255": dict(field="body", claim="Integration with threat modeling, attack tree, mitigations, and test case generation",
        situation="S2", provenance="replica", expected="refuted", alternatives=["insufficient_evidence"],
        minimal_evidence=["main.py: única rama `elif model_provider == \"Amazon Bedrock\":` → get_threat_model_bedrock(...)",
                          "threat_model.py añade get_threat_model_bedrock; attack_tree.py y mitigations.py no cambian; las líneas de import de attack_tree/mitigations (contexto, sin cambios) no incluyen función Bedrock",
                          "trampa: test_cases.py (módulo de generación de casos Gherkin) recibe +227 líneas que son unittest de Bedrock para threat model, no integración de generación de casos"],
        reasoning="Solo la ruta de threat modeling tiene integración Bedrock; las otras tres funciones no la tienen. Citar test_cases.py como prueba de 'test case generation' es una premisa homónima inválida."),
    "3152317896": dict(field="body", claim="Implemented `porto create-merchant` command for creating merchant accounts with secp256k1 keys",
        situation="S2", provenance="replica", expected="refuted", alternatives=[],
        minimal_evidence=["src/cli/index.ts: cac('porto'); únicos comandos registrados '[root]' y 'create-account [alias: ca]' ('Create a Porto Account')",
                          "commands.ts: createAccount usa Key.createSecp256k1()"],
        reasoning="El CLI registra create-account, no create-merchant: contraejemplo concreto. La parte secp256k1 es cierta pero no salva el nombre del comando. 'insufficient_evidence' sería abstención en un caso resoluble."),
    "3214876564": dict(field="body", claim="Uses SHA-256 hash of plan content for consistent keys",
        situation="S2", provenance="replica", expected="refuted", alternatives=["insufficient_evidence"],
        minimal_evidence=["runtime.ts: `const memoryKey = `action_${responseAction}_${uuidv4()}`` (UUID v4 aleatorio)",
                          "no hay sha/createHash/crypto en ninguna línea añadida",
                          "trampa: `import { createUniqueUuid }` es contexto de base sin cambios"],
        reasoning="La única generación de claves añadida es aleatoria. 'insufficient_evidence' es aceptable si se argumenta que el id del plan podría generarse fuera del diff. Usar createUniqueUuid como prueba sería premisa de base inválida."),
    "2886171819": dict(field="body", claim="The fix ensures this_order is always at least 1",
        situation="S1", provenance="extension", expected="derived", alternatives=["insufficient_evidence"],
        minimal_evidence=["+ this_order = max(1, min(self.config.solver_order, len(self.timesteps) - self.step_index)) en la rama lower_order_final",
                          "rama else: this_order = self.config.solver_order (sin cambios)"],
        reasoning="max(1, x) ≥ 1 para todo x en la rama corregida. 'insufficient_evidence' es aceptable si señala que en la rama else el valor depende de solver_order, no visible en la evidencia."),
    "2964743634": dict(field="title", claim="Add TTY environment check to `cargo insta review`",
        situation="S1", provenance="extension", expected="derived", alternatives=["insufficient_evidence"],
        minimal_evidence=["cargo-insta/src/cli.rs, fn query_snapshot: + if !term.is_term() { return Err(err_msg(\"Interactive review requires a terminal...\")) }"],
        reasoning="La comprobación TTY se demuestra en el código. El vínculo con `cargo insta review` debe apoyarse en el código (función del bucle interactivo y su mensaje), no solo en la línea de CHANGELOG, que es texto del autor. 'insufficient_evidence' es aceptable si señala que la ruta de llamada desde el comando review no está en la evidencia."),
    "3037457814": dict(field="body", claim="This PR adds the Language Server (https://github.com/isaacphi/mcp-language-server) to the code analysis & quality category.",
        situation="S1", provenance="extension", expected="derived", alternatives=[],
        minimal_evidence=["+ - [isaacphi/mcp-language-server](https://github.com/isaacphi/mcp-language-server): ...", "en docs/code-analysis--quality.md, cuya cabecera describe servidores de análisis estático/calidad"],
        reasoning="Enlaza la línea añadida con el archivo/categoría."),
    "3197644488": dict(field="title", claim="docs: add release notes for v0.0.4",
        situation="S1", provenance="extension", expected="derived", alternatives=[],
        minimal_evidence=["archivo añadido docs/release-notes/v0.0.4.md con '# Release v0.0.4'"],
        reasoning="Archivo nuevo de notas de versión con la versión indicada."),
    "3246950577": dict(field="body", claim="This resolves an issue where running `pdm export --lockfile pylock.toml` would produce empty requirements.txt files.",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=["derived"],
        minimal_evidence=["export.py: candidatos vía evaluate_candidates(groups, not options.markers); se elimina el filtro `groups.isdisjoint(...)`",
                          "lock.py: evaluate_candidates(..., evaluate_markers=True)",
                          "tests/test_formats.py: test_export_from_pylock_not_empty afirma salida no vacía con -L pylock.toml"],
        reasoning="La evidencia muestra el cambio de filtrado y un test de regresión, pero no su ejecución. 'derived' es aceptable solo si encadena el cambio de export.py con el test y declara como supuesto que el test pasa."),
    "3245466752": dict(field="body", claim="This PR fixes the output to include the server name as the root key",
        situation="S1", provenance="extension", expected="derived", alternatives=[],
        minimal_evidence=["mcp_json.py: + config = {name: server_config}; json_output = json.dumps(config, indent=2)",
                          "name procede de server_name (`--name`, -n) vía install_mcp_json(name=name)"],
        reasoning="Encadena el envoltorio {name: ...} con la serialización de salida. La documentación cambiada es texto del autor, no prueba de comportamiento. Trampa: archivo renombrado desde mcp_config.py."),
    "3119417980": dict(field="title", claim="Add includeStackTrace option to reduce LLM token usage by 80-90%",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=[],
        minimal_evidence=["opción includeStackTrace en el schema zod, toolHandler (default true) y Unity (GetBoolParameter, GetLogsAsJson omite stackTrace si false)",
                          "'80-90%' solo aparece en cadenas descriptivas escritas por el autor dentro del diff; no hay medición"],
        reasoning="La versión débil (se añadió la opción y elimina el stack trace) se sostiene. La magnitud 80-90% no tiene evidencia independiente. 'derived' sería falsa aceptación por usar texto del autor como premisa."),
    "3047699666": dict(field="body", claim="Added test file `frontend/packages/db/supabase/tests/database/03-organization_members_rls.test.sql` with 8 test cases",
        situation="S1", provenance="extension", expected="derived", alternatives=[],
        minimal_evidence=["archivo añadido con esa ruta", "+SELECT plan(8);", "8 bloques '-- Test N' y 8 aserciones `SELECT ok(`"],
        reasoning="Recuento: plan(8) y 8 aserciones coinciden con el 8 afirmado."),
    "2986431376": dict(field="body", claim="Fix heading hierarchy to follow consistent H1 -> H2 -> H3 structure",
        situation="S3", provenance="extension", expected="insufficient_evidence", alternatives=[],
        minimal_evidence=["cambios ### → ## y #### → ### en kubernetes.md", "el inventario completo de cabeceras del documento (incluido el H1) no está en la evidencia"],
        reasoning="Se sostiene 'se subió un nivel a varias cabeceras'. 'Consistent H1 -> H2 -> H3' en todo el documento no es determinable con los fragmentos del diff."),
    "3196885081": dict(field="body", claim="mcp/: Server registry & orchestration (207 lines total)",
        situation="S1", provenance="extension", expected="derived", alternatives=["insufficient_evidence"],
        minimal_evidence=["patterns/mcp/registry.yml añadido: @@ -0,0 +1,77 @@, sin salto final", "patterns/mcp/orchestration.yml añadido: @@ -0,0 +1,132 @@, sin salto final"],
        reasoning="77 + 132 = 209 líneas de contenido; ambos archivos terminan sin salto de línea, por lo que `wc -l` da 76 + 131 = 207. 'insufficient_evidence' es aceptable si se señala la ambigüedad del convenio de conteo. Refutar con 209 (o 234 incluyendo index.yml) es una refutación incorrecta."),
    "3135266974": dict(field="body", claim="Implements full `Learner` protocol (`partial_fit`, `sample`, `predict`, `decay`)",
        situation="S1", provenance="extension", expected="derived", alternatives=["insufficient_evidence"],
        minimal_evidence=["pipelines/_learner.py (añadido): class LearnerPipeline con def sample, def partial_fit, def decay, def predict"],
        reasoning="Enlaza cuatro métodos definidos en la misma clase. 'insufficient_evidence' es aceptable si señala que la definición del protocolo Learner no está en la evidencia ('full'). Trampa: _agent.py viene renombrado desde _agent_pipeline.py."),
}

# Cases whose reference verdict needs >=2 linked facts or a general rule (S1 except the
# single-fact release-notes case, plus all S2 counterexamples). S3 cases test scope discipline.
REQUIRES_INFERENCE = {"3069923749", "3204772289", "2889875255", "3152317896", "3214876564", "2886171819",
                      "2964743634", "3037457814", "3245466752", "3047699666", "3196885081", "3135266974"}

ROUTING = {"derived": "PASS", "refuted": "RETRY", "insufficient_evidence": "ESCALATE"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def build_sources(meta: dict, files: list[dict]) -> list[dict]:
    lines = [
        f"Pull request diff summary (base commit {meta['base']['sha']} -> head commit {meta['head']['sha']})",
        f"Files changed: {meta.get('changed_files', len(files))}",
        f"Lines added: {meta.get('additions', 0)}",
        f"Lines deleted: {meta.get('deletions', 0)}",
        "Changed files:",
    ]
    if not files:
        lines.append("(none)")
    for f in files:
        ren = f", renamed from {f['previous_filename']}" if f.get("previous_filename") else ""
        lines.append(f"- {f['filename']} ({f['status']}{ren}, +{f['additions']}/-{f['deletions']})")
    sources = [{"document_id": "diff_summary", "text": "\n".join(lines)}]
    for i, f in enumerate(files, 1):
        ren = f", renamed from {f['previous_filename']}" if f.get("previous_filename") else ""
        text = (f"File: {f['filename']} (status: {f['status']}{ren})\n"
                f"Unified diff from base to head ('-' = base only, '+' = head only, ' ' = unchanged context):\n"
                f"{f['patch']}")
        sources.append({"document_id": f"file_{i:02d}", "text": text})
    return sources


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main(force: bool = False) -> None:
    out_in, out_exp = FROZEN / "inputs_manifest.json", FROZEN / "expected_manifest.json"
    if (out_in.exists() or out_exp.exists()) and not force:
        sys.exit("Frozen manifests already exist; refusing to overwrite (use --force before the first call only).")
    sel = json.loads((HERE / "selection_log.json").read_text(encoding="utf-8"))
    val = {r["pr_id"]: r for r in csv.DictReader(open(PRMCI / "validation/validation_results.csv", encoding="utf-8-sig"))}
    prs = {p["pr_id"]: p for p in parse.load("Claude_Code", prmci_dir=str(PRMCI))}
    selected = sorted(s["pr_id"] for s in sel["selected"])
    assert len(selected) == 20 and set(selected) == set(CASES), (set(selected) ^ set(CASES))
    inputs, expected = [], []
    for idx, pid in enumerate(selected, 1):
        cid = f"C{idx:02d}"
        c, p, v = CASES[pid], prs[pid], val[pid]
        src_text = p["title"] if c["field"] == "title" else p["body"]
        assert norm(c["claim"]) in norm(src_text), (pid, c["claim"])
        meta = json.loads((HERE / "github_cache" / f"{pid}_pull.json").read_text(encoding="utf-8"))
        fpath = HERE / "github_cache" / f"{pid}_files.json"
        files = json.loads(fpath.read_text(encoding="utf-8")) if fpath.exists() else []
        sources = build_sources(meta, files)
        s_entry = next(s for s in sel["selected"] if s["pr_id"] == pid)
        inputs.append({
            "case_id": cid, "pr_id": pid, "url": p["url"], "repository": "/".join(p["url"].split("/")[3:5]),
            "base_sha": meta["base"]["sha"], "head_sha": meta["head"]["sha"],
            "merged": meta.get("merged"), "merge_commit_sha": meta.get("merge_commit_sha"),
            "claim": c["claim"], "claim_source_field": c["field"], "claim_author": "Claude Code (PR author agent)",
            "evidence_origin": "GitHub REST pulls/{n}/files (net base->head diff), fetched 2026-11-21, cached in github_cache/",
            "corpus_patch_sha256": sha256_text(p["patch"] or ""),
            "net_patch_sha256": sha256_text("\n".join(f.get("patch", "") for f in files)),
            "corpus_vs_net_stats_issues": s_entry.get("post_check_issues", []),
            "sources": sources,
        })
        expected.append({
            "case_id": cid, "pr_id": pid, "situation": c["situation"], "provenance": c["provenance"],
            "expected_verdict": c["expected"], "expected_routing": ROUTING[c["expected"]],
            "acceptable_alternative_verdicts": c["alternatives"],
            "minimal_evidence": c["minimal_evidence"], "acceptable_reasoning": c["reasoning"],
            "requires_inference": pid in REQUIRES_INFERENCE,
            "prmci_gold_pr_label": (v["consensus_label"] or v["annotator1_label"]).lower(),
            "prmci_annotator_notes": {"a1": v["annotator1_notes"], "a2": v["annotator2_notes"], "consensus": v["consensus_notes"]},
            "gold_transferability": ("not_transferable_corpus_vs_net_diff" if pid in ("2975165913", "3146561196") else "transferable"),
        })
    FROZEN.mkdir(exist_ok=True)
    out_in.write_text(json.dumps({"study": "PR-MCI Claude_Code claim derivation eval", "prmci_commit": "ac6bbf4deab044d86389a6e178441292306bedcf",
                                  "cases": inputs}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out_exp.write_text(json.dumps({"verdict_names": "claimlayer.v4 epistemic_verdict: derived | refuted | insufficient_evidence (unresolved = system could not decide)",
                                   "cases": expected}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    h_in, h_exp = sha256_text(out_in.read_text(encoding="utf-8")), sha256_text(out_exp.read_text(encoding="utf-8"))
    (FROZEN / "MANIFEST_HASHES.txt").write_text(f"{h_in}  inputs_manifest.json\n{h_exp}  expected_manifest.json\n", encoding="utf-8")
    print("inputs", h_in); print("expected", h_exp)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
