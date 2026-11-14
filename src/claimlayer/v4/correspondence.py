"""Response-to-Graph Correspondence & Claim-Level Semantic Repair Verification.

Verifies whether a natural-language candidate response genuinely aligns with an
explicit JustificationGraph rather than passing surface keyword checks:
- Safe AST arithmetic equation parser & graph-step correspondence verifier
  (``FALSE_EQUATION_IN_RESPONSE``, ``UNGROUNDED_EQUATION_IN_RESPONSE``,
  ``UNVERIFIED_EXPRESSION_IN_RESPONSE``).
- Structured condition & polarity comparator distinguishing exceptions
  (``salvo``, ``a menos que``), prerequisites (``si``, ``siempre que``), and
  concessive overrides (``incluso en caso de``, ``aunque``), plus unrelated
  condition substitutions and unresolved condition semantics.
- Claim-by-claim clause decomposition checks conclusion polarity
  (``CONCLUSION_POLARITY_INVERTED``) and flags ungrounded non-numeric factual
  additions (e.g. ``"Warren Buffet es astronauta"``) via a normalized grounding
  lexicon (and optional ``SemanticEvaluatorConfig`` fallback).
- Compositional structural patterns for source attribution, numerical
  estimation/approximation, and missing calendar granularity so valid
  paraphrases are accepted without brittle phrase allowlists.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence
import unicodedata

from ..semantic.models import SourceDocument, digest
from .justification import JustificationGraph
from .vocabulary import UnitRegistry


@dataclass(frozen=True)
class EquationCheckResult:
    """Result of parsing and verifying a mathematical equality in text."""

    raw_equation: str
    lhs_expr: str
    rhs_expr: str
    lhs_value: Decimal | None
    rhs_value: Decimal | None
    is_mathematically_true: bool | None
    matches_graph_step: bool
    matched_step_id: str | None
    status: str  # "verified" | "refuted" | "unknown"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_equation": self.raw_equation,
            "lhs_expr": self.lhs_expr,
            "rhs_expr": self.rhs_expr,
            "lhs_value": str(self.lhs_value) if self.lhs_value is not None else None,
            "rhs_value": str(self.rhs_value) if self.rhs_value is not None else None,
            "is_mathematically_true": self.is_mathematically_true,
            "matches_graph_step": self.matches_graph_step,
            "matched_step_id": self.matched_step_id,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StructuredCondition:
    """Structured condition extracted from graph requirements or response text."""

    raw_text: str
    kind: str  # "exception" | "prerequisite" | "concessive_override" | "unmarked"
    effective_polarity: str  # "excluding" | "requiring" | "concessive_override" | "unmarked"
    content_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "kind": self.kind,
            "effective_polarity": self.effective_polarity,
            "content_tokens": list(self.content_tokens),
        }


@dataclass(frozen=True)
class ResponseClauseClaim:
    """Single clause-level claim decomposed from a natural-language response."""

    clause_text: str
    role: str
    subject: str | None
    polarity: str  # "affirmative" | "negative"
    numbers: tuple[Decimal, ...]
    is_grounded: bool
    ungrounded_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_text": self.clause_text,
            "role": self.role,
            "subject": self.subject,
            "polarity": self.polarity,
            "numbers": [str(n) for n in self.numbers],
            "is_grounded": self.is_grounded,
            "ungrounded_tokens": list(self.ungrounded_tokens),
        }


_UNICODE_MINUS_RE = re.compile(r"[−–—]")

_APPROX_MODIFIER_RE = re.compile(
    r"(?:\b(?:aproximad[oa]s?|aproximadamente|aprox\.?|alrededor\s+de|cerca\s+de|"
    r"en\s+torno\s+a|aproximaci[oó]n|estimaci[oó]n|estimad[oa]s?|se\s+estima|"
    r"estima(?:\s+en)?|ronda(?:\s+los|\s+las)?|rondar[ií]a|oscila|m[aá]s\s+o\s+menos|"
    r"tendr[ií]a|podr[ií]a\s+tener|estar[ií]a|habr[ií]a\s+nacido|"
    r"unos\s+(?=\d)|unas\s+(?=\d)|"
    r"approximately|approx\.?|roughly(?:\s+speaking)?|around|about|estimated?)\b|"
    r"~\s*(?=\d))",
    re.IGNORECASE,
)

_RANGE_PATTERN_RE = re.compile(
    r"(?:\b(?:entre|between|de|from)\s+(-?\d+(?:[.,]\d+)?)\s+(?:y|a|and|to)\s+(-?\d+(?:[.,]\d+)?)\b|"
    r"\[\s*(-?\d+(?:[.,]\d+)?)\s*[-,]\s*(-?\d+(?:[.,]\d+)?)\s*\]|"
    r"\b(-?\d+(?:[.,]\d+)?)\s*[–]\s*(-?\d+(?:[.,]\d+)?)\b|"
    r"(-?\d+(?:[.,]\d+)?)\s*(?:±|\+/-)\s*(\d+(?:[.,]\d+)?))",
    re.IGNORECASE,
)

_EXACT_ASSERTION_RE = re.compile(
    r"\b(exactamente|exacta|exacto|exact|exactly|precisamente|precisely|"
    r"con exactitud|de forma exacta|sabemos que tiene exactamente|sin lugar a dudas|"
    r"inequ[ií]vocamente|indiscutiblemente|definitivamente|total completo|total general)\b",
    re.IGNORECASE,
)

_NEGATED_LIMIT_CONTEXT_RE = re.compile(
    r"\b(no se puede|no es posible|imposible|no hay|no existen?|no se conocen?|no se sabe|"
    r"no constan?|al no constar|al no tener|no basta|no alcanza|impide|sin|faltan?|falta de|"
    r"carece de|ausencia de|no permite|no puede afirmarse|para precisar|"
    r"no es exact[ao]|no puede determinarse|cannot|can't|not possible|unable to|without|"
    r"missing|lacking|insufficient|not verified as|not exact|cannot determine|impossible to determine)\b",
    re.IGNORECASE,
)

_DEFICIT_CUE = (
    r"(?:sin|faltan?|falta\s+de|no\s+hay|no\s+constan?|al\s+no\s+constar|al\s+no\s+tener|"
    r"no\s+se\s+conocen?|se\s+desconocen?|carece\s+de|ausencia\s+de|no\s+se\s+especifican?|"
    r"without|missing|lacking)"
)
_GRANULARITY_TERM = (
    r"(?:(?:el|la|los|las|de|the|full|exact)\s+)*"
    r"(?:fechas?\s+completas?|fechas?\s+exactas?|fecha\s+de\s+nacimiento|"
    r"d[ií]a(?:\s+(?:y|ni)\s+(?:el\s+)?mes)?|mes(?:\s+(?:y|ni)\s+(?:el\s+)?d[ií]a)?|"
    r"granularidad|fecha\s+parcial|day\s+and\s+month|month\s+and\s+day|"
    r"completitud|poblaci[oó]n\s+incompleta)"
)
_MISSING_PRECISION_DISCLOSURE_RE = re.compile(
    rf"\b(?:{_DEFICIT_CUE}\s+{_GRANULARITY_TERM}|"
    r"solo\s+se\s+conoce\s+el\s+a[nñ]o|"
    r"poblaci[oó]n\s+incompleta|sin\s+constancia\s+de\s+completitud|incomplete\s+population|"
    r"no\s+representan\s+necesariamente\s+el\s+total)\b",
    re.IGNORECASE,
)

_ATTRIB_PREP = (
    r"(?:seg[uú]n(?:\s+(?:se\s+desprende|consta|indica|figura)\s+(?:en|de(?:l)?))?"
    r"|a\s+partir\s+de(?:l)?|con\s+base\s+en|bas[aá]ndose\s+en|de\s+acuerdo\s+con"
    r"|conforme\s+a(?:l)?|a\s+tenor\s+de(?:l)?|atendiendo\s+a(?:l)?"
    r"|como\s+(?:indica|se[nñ]ala|recoge|menciona)\s+(?:el|la|los|las)?"
    r"|according\s+to|based\s+on|from|per|in\s+light\s+of|as\s+(?:stated|noted|indicated)\s+in)"
)
_ATTRIB_SOURCE_NOUN = (
    r"(?:(?:el|la|los|las|este|esta|dicho|dicha|lo|the|supplied|provided|given|present)\s+)*"
    r"(?:textos?|pasajes?|documentos?|extractos?|fragmentos?|fuentes?|evidencias?|"
    r"facturas?|datos|informaci[oó]n|relatos?|aportad[oa]s?|facilitad[oa]s?|"
    r"proporcionad[oa]s?|suministrad[oa]s?|passages?|texts?|documents?|excerpts?|sources?|evidence)"
)
_SOURCE_ATTRIBUTION_RE = re.compile(
    rf"\b(?:{_ATTRIB_PREP}\s+{_ATTRIB_SOURCE_NOUN}|"
    r"no\s+se\s+ha\s+comprobado.*externamente)\b",
    re.IGNORECASE,
)

_CONTRADICTORY_EXTERNAL_TRUTH_RE = re.compile(
    r"\b(hecho hist[oó]rico verificado|comprobado externamente|contrastado externamente|"
    r"verdad emp[ií]rica demostrada|independently verified historical fact|"
    r"externally confirmed beyond doubt|proven externally)\b",
    re.IGNORECASE,
)

_SUBTOTAL_QUALIFICATION_RE = re.compile(
    r"\b(subtotal|suma parcial|parcial|de los elementos aportados|de las facturas listadas|"
    r"partial sum|listed items only|subtotal only)\b",
    re.IGNORECASE,
)

_UNQUALIFIED_TOTAL_RE = re.compile(
    r"\b(el total es|total general|total completo|monto total definitivo|total amount is|full total|grand total)\b",
    re.IGNORECASE,
)

_CONFLICT_DISCLOSURE_RE = re.compile(
    r"\b(conflicto|contradicci[oó]n|contradictorio|discrepancia|fuentes divergentes|"
    r"sin resolver|conflict|contradict|discrepancy|disagree|conflicting sources|"
    r"divergent|inconclusive)\b",
    re.IGNORECASE,
)

_CONCESSIVE_OVERRIDE_RE = re.compile(
    r"\b(incluso\s+en\s+caso\s+de|incluso\s+si|incluso\s+con|incluso\s+ante|"
    r"aunque(?:\s+haya|\s+exista)?|a\s+pesar\s+de|pese\s+a|independientemente\s+de|"
    r"con\s+o\s+sin|even\s+if|even\s+in\s+case\s+of|despite|regardless\s+of|whether\s+or\s+not)\b",
    re.IGNORECASE,
)

_EXCEPTION_MARKER_RE = re.compile(
    r"\b(salvo(?:\s+en\s+caso\s+de|\s+que)?|excepto(?:\s+en\s+caso\s+de|\s+que|\s+si)?|"
    r"a\s+menos\s+que|a\s+excepci[oó]n\s+de|con\s+excepci[oó]n\s+de|sin\s+que\s+(?:exista|haya)|"
    r"solo\s+si\s+no|siempre\s+que\s+no|si\s+no\s+(?:hay|existe)|no\s+aplica\s+(?:si|en\s+caso\s+de)|"
    r"ausencia\s+de|unless|except(?:\s+in\s+case\s+of|\s+when|\s+if)?|"
    r"provided\s+there\s+is\s+no|only\s+if\s+no)\b",
    re.IGNORECASE,
)

_PREREQUISITE_MARKER_RE = re.compile(
    r"\b(si\b(?!\s+no\b)|siempre\s+que\b(?!\s+no\b)|solo\s+si\b(?!\s+no\b)|"
    r"en\s+caso\s+de|cuando|condicionad[oa]\s+a|sujet[oa]\s+a|bajo\s+la\s+condici[oó]n\s+de|"
    r"if\b(?!\s+there\s+is\s+no\b)|provided\s+that|only\s+if\b(?!\s+no\b)|"
    r"subject\s+to|conditional\s+upon)\b",
    re.IGNORECASE,
)

_CONDITIONAL_MARKER_RE = re.compile(
    r"\b(si|salvo|siempre que|a menos que|condicionad[oa]|sujet[oa] a|excepto|"
    r"incluso en caso de|incluso si|aunque|a pesar de|"
    r"if|unless|subject to|provided that|conditional upon|contingent on|except|even if|despite)\b",
    re.IGNORECASE,
)

_CONCLUSION_NEGATION_RE = re.compile(
    r"(?:\b(?:no|nunca|jam[aá]s|tampoco|not|never)\s+"
    r"(?:es|son|era|eran|fue|fueron|ser[ií]a|ser[aá]|tiene|tienen|ten[ií]a|tendr[ií]a|tuvo|"
    r"alcanza|asciende|equivale|corresponde|aplica|supera|llega|se\s+estima|se\s+sit[uú]a|"
    r"representa|constituye|"
    r"is|are|was|were|would|will|has|have|had|does|do|did|equals|reaches|applies)\b"
    r"|\b(?:does\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|would\s+not|"
    r"will\s+not|has\s+not|have\s+not)\b"
    r"|\bno\s+(?:de|aprox\.?|aproximadamente|unos|unas|entre)\s+(?=\d))",
    re.IGNORECASE,
)

_SAFE_DISCLAIMER_NEGATION_RE = re.compile(
    r"\b(?:no\s+(?:se\s+puede|es\s+posible|ha\s+sido\s+verificad[oa]|est[aá]\s+verificad[oa]|"
    r"se\s+ha\s+comprobado|hay|constan?|se\s+conocen?|representan?\s+necesariamente)|"
    r"sin\s+(?:verificar|constancia|fechas)|not\s+(?:verified|confirmed|possible)|cannot\s+determine)\b",
    re.IGNORECASE,
)

_EVASION_NO_ANSWER_RE = re.compile(
    r"\b(no puedo responder|no es posible responder|prefiero no responder|"
    r"no dar[eé] una respuesta|no se aborda|omito responder|"
    r"cannot answer the question|will not answer|unable to provide any estimate)\b",
    re.IGNORECASE,
)

_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.-])(-?\d+(?:[.,]\d+)?)(?![A-Za-z0-9_.-])")
_YEAR_OR_DATE_TOKEN_RE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?|(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\b")

_UNIT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "years": ("año", "años", "year", "years", "yr", "yrs"),
    "months": ("mes", "meses", "month", "months"),
    "calendar_days": ("día", "días", "day", "days", "calendar days", "días naturales"),
    "business_days": ("días hábiles", "días laborables", "business day", "business days", "working days"),
    "USD": ("usd", "dólar", "dólares", "dollar", "dollars", "$"),
    "EUR": ("eur", "euro", "euros", "€"),
    "GBP": ("gbp", "libra", "libras", "pound", "pounds", "£"),
    "percent": ("%", "por ciento", "percent", "pct"),
    "hours": ("hora", "horas", "hour", "hours", "hr", "hrs"),
    "minutes": ("minuto", "minutos", "minute", "minutes", "min"),
}

_FUNCTION_WORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por", "para",
    "que", "se", "su", "sus", "al", "es", "son", "era", "eran", "fue", "fueron", "ser", "seria", "sera",
    "esta", "estan", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "esto", "eso",
    "estamos", "hay", "existe", "existen", "existia", "existian", "ha", "han", "haya", "habia", "habria", "sido", "siendo", "tiene", "tienen",
    "tenia", "tendria", "tuvo", "puede", "pueden", "podria", "debe", "deben", "sabe", "sabemos",
    "y", "o", "u", "ni", "si", "no", "sin", "sobre", "entre", "desde", "hasta", "como", "cual", "quien",
    "cuando", "donde", "pero", "sino", "aunque", "debido", "dado", "dada", "ambas", "ambos", "otra",
    "otro", "otras", "otros", "toda", "todo", "todas", "todos", "cada", "mas", "menos", "muy", "tan",
    "solo", "sola", "solamente", "unicamente", "ademas", "embargo", "obstante", "entonces", "luego",
    "tanto", "mientras", "pesar", "pese", "respecto", "incluso", "durante", "mediante", "ante", "bajo",
    "hacia", "tras", "porque", "pues",
    "the", "and", "for", "with", "that", "this", "these", "those", "from", "into", "when", "where",
    "which", "who", "was", "were", "is", "are", "be", "been", "being", "has", "have", "had", "would",
    "could", "should", "may", "might", "can", "cannot", "not", "without", "within", "what", "how",
    "many", "much", "both", "each", "all", "any", "other", "given", "only", "solely", "however",
    "therefore", "thus", "hence", "but", "yet", "while", "whereas", "unless", "except", "despite",
    "even", "though", "than", "then",
})

_COMMON_STOPWORDS = _FUNCTION_WORDS | frozenset({
    "segun", "según", "texto", "documento", "calculo", "cálculo", "aproximadamente", "edad",
    "anos", "años", "ano", "año", "fecha", "fechas", "completas", "exacta", "exacto",
    "subtotal", "total", "condicion", "condición", "salvo", "fuentes", "conflicto",
    "about", "approx", "approximately", "according", "text", "document", "year", "years",
    "age", "old", "exact", "dates", "full", "based", "note", "calculation", "partial",
})

_CONDITION_SYNTAX_WORDS = _FUNCTION_WORDS | frozenset({
    "caso", "casos", "salvo", "excepto", "excepcion", "condicion", "sujeto", "sujeta",
    "condicionado", "condicionada", "aplica", "procede", "existe", "exista", "existen",
    "ausencia", "incluso", "pesar", "independientemente", "case", "cases", "unless",
    "except", "exception", "condition", "subject", "provided", "applies", "exists",
})

# Closed structural/epistemic vocabulary used to formulate verified answers and limitations
# without introducing external factual claims.
_EPISTEMIC_META_TOKENS = frozenset({
    # Spanish attribution / source references
    "segun", "partir", "pasaje", "pasajes", "texto", "textos", "documento", "documentos",
    "extracto", "extractos", "fragmento", "fragmentos", "fuente", "fuentes", "evidencia",
    "evidencias", "datos", "informacion", "relato", "aportado", "aportada", "aportados",
    "aportadas", "facilitado", "facilitada", "facilitados", "facilitadas", "proporcionado",
    "proporcionada", "suministrado", "suministrada", "citado", "citada", "acuerdo", "conforme",
    "tenor", "atendiendo", "basandose", "desprende", "consta", "constan", "figura", "indica",
    "indican", "senala", "senalan", "recoge", "menciona", "establece",
    # Spanish epistemic / approximation / granularity / derivation
    "conclusion", "estimacion", "estima", "estimado", "estimada", "estimar", "aproximacion",
    "aproximadamente", "aproximado", "aproximada", "aprox", "alrededor", "torno", "cerca",
    "ronda", "rondaria", "oscila", "exacta", "exacto", "exactos", "exactas", "exactamente", "exactitud",
    "precisa", "preciso", "precisar", "precisarse", "hoy", "ahora", "actualmente", "certeza", "duda", "dudas", "inequivocamente",
    "indiscutiblemente", "definitivamente", "determinar", "determinarse", "concluir", "afirmar",
    "afirmarse", "verificar", "verificado", "verificada", "verificados", "comprobar", "comprobado",
    "comprobada", "contrastar", "contrastado", "contrastada", "externo", "externa", "externos",
    "externamente", "independiente", "independientemente", "historico", "historica",
    "historicamente", "hecho", "depende", "meramente", "fecha", "fechas", "completa", "completo",
    "completas", "completos", "nacimiento", "cumpleanos", "granularidad", "parcial", "parciales",
    "dia", "dias", "mes", "meses", "ano", "anos", "faltan", "falta", "carece", "ausencia",
    "desconocen", "desconoce", "especifican", "especifica", "calculo", "calculado", "calculada",
    "formula", "ecuacion", "operacion", "resultado", "valor", "cifra", "cifras", "unica", "unico",
    "situaria", "situa", "razon", "posible", "imposible", "basta", "alcanza", "asciende",
    "equivale", "corresponde", "aplica", "representa", "represente", "representan",
    "necesariamente",
    # Spanish aggregation / condition / conflict qualifiers
    "subtotal", "total", "suma", "partidas", "elementos", "incluidas", "incluidos", "listadas",
    "listados", "constancia", "completitud", "poblacion", "general", "condicion", "excepcion",
    "salvedad", "caso", "casos", "salvo", "excepto", "sujeto", "sujeta", "condicionado",
    "condicionada", "conflicto", "contradiccion", "contradictorio", "discrepancia", "divergentes",
    "resolver", "nota", "aclaracion", "limitacion", "limitaciones", "combinado", "combinada",
    "responder", "pregunta", "omito", "abordar", "aborda",
    # English counterparts
    "according", "passage", "passages", "excerpt", "excerpts", "text", "texts", "document",
    "documents", "source", "sources", "evidence", "supplied", "provided", "facilitated", "stated",
    "noted", "indicated", "indicates", "mentions", "establishes", "based", "depends", "solely",
    "conclusion", "estimate", "estimated", "estimation", "approximate", "approximately", "approx",
    "roughly", "around", "about", "exact", "exactly", "precision", "precise", "certainty",
    "determine", "verify", "verified", "confirmed", "external", "externally", "independently",
    "historical", "fact", "empirical", "date", "dates", "full", "complete", "completeness",
    "birth", "birthday", "granularity", "partial", "partially", "day", "days", "month", "months",
    "year", "years", "missing", "lacking", "calculation", "formula", "equation", "result",
    "value", "figure", "single", "unique", "possible", "impossible", "necessarily", "subtotal",
    "total", "sum", "items", "listed", "included", "population", "condition", "exception",
    "subject", "conditional", "contingent", "conflict", "contradiction", "discrepancy",
    "divergent", "unresolved", "inconclusive", "combined", "answer", "question", "today", "now", "currently",
})

_CONJ_VERB_SPLIT_RE = re.compile(
    r"\s+(?:y|e|and)\s+(?=(?:es|son|era|fueron|fue|est[aá]|est[aá]n|tiene|tienen|trabaja|vive|"
    r"naci[oó]|muri[oó]|falleci[oó]|posee|ostenta|act[uú]a|participa|dirige|"
    r"is|are|was|were|has|have|works|lives|born|died|owns|leads)\b)",
    re.IGNORECASE,
)

_DISCOURSE_SPLIT_RE = re.compile(
    r",\s+(?=(?:aunque|pero|mas|sino|sin\s+embargo|no\s+obstante|si\s+bien|mientras\s+que|"
    r"although|though|even\s+though|however|whereas|while)\b)",
    re.IGNORECASE,
)


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_token(tok: str) -> str:
    return _strip_accents(tok.lower().strip())


def _stem_key(norm_tok: str) -> str:
    """Return a conservative morphological key only for longer content words (>= 6 chars)."""
    if len(norm_tok) < 6:
        return norm_tok
    if norm_tok.endswith(("aciones", "iciones", "amiento")):
        return norm_tok[:-4]
    if norm_tok.endswith(("mente", "acion", "icion", "ables", "ibles")):
        return norm_tok[:-4]
    if norm_tok.endswith(("ados", "adas", "idos", "idas")):
        return norm_tok[:-3]
    if norm_tok.endswith(("ado", "ada", "ido", "ida", "ces", "es")):
        return norm_tok[:-2] if len(norm_tok) >= 7 else norm_tok
    if norm_tok.endswith(("s", "a", "o", "e")):
        return norm_tok[:-1]
    return norm_tok


def compute_response_digest(
    candidate_response: str | None,
    question: str | Mapping[str, Any] | None = None,
) -> str | None:
    """Compute canonical digest binding exact response text and question."""
    if candidate_response is None:
        return None
    q_norm = (
        question
        if isinstance(question, str)
        else (str(question.get("text") or question.get("question") or "") if isinstance(question, Mapping) else "")
    )
    return digest({"question": q_norm, "candidate_response": candidate_response})


def _to_dec(raw: Any) -> Decimal | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, Decimal):
        return raw if raw.is_finite() else None
    try:
        s = _UNICODE_MINUS_RE.sub("-", str(raw).strip()).replace(",", ".")
        d = Decimal(s)
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _split_clauses(text: str) -> list[str]:
    norm = _UNICODE_MINUS_RE.sub("-", text or "")
    parts = re.split(
        r"(?:[.;!?\n]+|,\s*(?=(?:por tanto|por lo tanto|as[ií] que|pero|sin embargo|no obstante|therefore|hence|thus|so|however|but)\b))",
        norm,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _split_claim_subclauses(clause: str) -> list[str]:
    out: list[str] = []
    for disc_piece in _DISCOURSE_SPLIT_RE.split(clause):
        for piece in _CONJ_VERB_SPLIT_RE.split(disc_piece):
            cleaned = piece.strip(" ,;:")
            if cleaned:
                out.append(cleaned)
    return out


def _extract_clause_numbers(clause: str) -> list[Decimal]:
    out: list[Decimal] = []
    for m in _NUMBER_TOKEN_RE.finditer(clause):
        d = _to_dec(m.group(1))
        if d is not None:
            out.append(d)
    return out


def _extract_ranges(text: str) -> list[tuple[Decimal, Decimal]]:
    ranges: list[tuple[Decimal, Decimal]] = []
    for m in _RANGE_PATTERN_RE.finditer(text):
        g = m.groups()
        if g[0] is not None and g[1] is not None:
            a, b = _to_dec(g[0]), _to_dec(g[1])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[2] is not None and g[3] is not None:
            a, b = _to_dec(g[2]), _to_dec(g[3])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[4] is not None and g[5] is not None:
            a, b = _to_dec(g[4]), _to_dec(g[5])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[6] is not None and g[7] is not None:
            center, tol = _to_dec(g[6]), _to_dec(g[7])
            if center is not None and tol is not None and tol >= 0:
                ranges.append((center - tol, center + tol))
    return ranges


_POST_NEGATED_EXACT_RE = re.compile(
    r"^\s+(?:no\s+(?:se\s+puede|puede|es|son|resulta)|cannot|can't|is\s+not|are\s+not|not\s+(?:determinable|known|possible))\b",
    re.IGNORECASE,
)


def _clause_asserts_exactness(clause: str) -> bool:
    for m in _EXACT_ASSERTION_RE.finditer(clause):
        prefix = clause[: m.start()]
        suffix = clause[m.end() :]
        if _NEGATED_LIMIT_CONTEXT_RE.search(prefix) or _POST_NEGATED_EXACT_RE.match(suffix):
            continue
        return True
    return False


def _clause_is_explanatory_formula_or_disclaimer(clause: str) -> bool:
    low = clause.lower()
    if _NEGATED_LIMIT_CONTEXT_RE.search(low) or _MISSING_PRECISION_DISCLOSURE_RE.search(low):
        return True
    if re.search(r"(c[aá]lculo|formula|f[oó]rmula|calculation|compute|operaci[oó]n)\s+(?:es|is)\s+[-+\d(]", low):
        return True
    if "=" in clause and any(op in clause for op in ("+", "-", "*", "/", "×", "÷")):
        return True
    return False


# ---------------------------------------------------------------------------
# Safe AST equation parser & verifier
# ---------------------------------------------------------------------------

_UNIT_WORDS_FOR_EQUATION = sorted(
    {s for syns in _UNIT_SYNONYMS.values() for s in syns if s not in {"$"}},
    key=len,
    reverse=True,
)
_EQUATION_UNIT_RE = re.compile(
    r"(?:(?<!\w)(?:"
    + "|".join(re.escape(u) for u in _UNIT_WORDS_FOR_EQUATION if u.isalpha())
    + r")(?!\w)|[$€£%])",
    re.IGNORECASE,
)
_EQUATION_PROSE_LEAD_SPLIT_RE = re.compile(
    r"(?:\b(?:es|son|da|resulta|equivale\s+a|ser[ií]a|is|equals|gives|como|where|donde)\b|[:;,])",
    re.IGNORECASE,
)
_EQUATION_RHS_EXPR_RE = re.compile(
    r"^\s*((?:[-+*/×÷()\d.,\s$€£%]|\b(?:"
    + "|".join(re.escape(u) for u in _UNIT_WORDS_FOR_EQUATION if u.isalpha())
    + r")\b)+)"
)


def _eval_ast_node(node: ast.AST, leaves: list[Decimal]) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, leaves)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        d = Decimal(str(node.value))
        leaves.append(d)
        return d
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast_node(node.operand, leaves)
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.USub):
            return -v
        raise ValueError("Unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, leaves)
        right = _eval_ast_node(node.right, leaves)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero in expression")
            return left / right
        raise ValueError("Unsupported binary op")
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def _eval_arithmetic_side(expr_str: str) -> tuple[Decimal | None, tuple[Decimal, ...]]:
    cleaned = _UNICODE_MINUS_RE.sub("-", expr_str).strip()
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    cleaned = _EQUATION_UNIT_RE.sub(" ", cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None, ()
    try:
        tree = ast.parse(cleaned, mode="eval")
        leaves: list[Decimal] = []
        val = _eval_ast_node(tree, leaves)
        if not val.is_finite():
            return None, ()
        return val, tuple(leaves)
    except Exception:
        return None, ()


def _unclosed_open_paren_indices(expr: str) -> list[int]:
    stack: list[int] = []
    for idx, ch in enumerate(expr):
        if ch == "(":
            stack.append(idx)
        elif ch == ")" and stack:
            stack.pop()
    return stack


def _strip_leading_prose_prefix(raw_lhs: str) -> str:
    lead_segments = _EQUATION_PROSE_LEAD_SPLIT_RE.split(raw_lhs)
    lhs = (lead_segments[-1] if lead_segments else raw_lhs).strip()
    unit_lows = {u.lower() for u in _UNIT_WORDS_FOR_EQUATION}
    while lhs:
        m = re.match(r"^([A-Za-zÀ-ÿ]+)\s+(?=.+)", lhs)
        if not m:
            break
        word = m.group(1)
        if word.lower() in unit_lows:
            break
        # Preserve single-letter algebraic variables preceding an arithmetic operator (e.g., "x + 3 = 5")
        rest = lhs[m.end():]
        if len(word) == 1 and re.match(r"^[+\-*/×÷]", rest):
            break
        lhs = rest.strip()
    return lhs


def _extract_equation_chains(text: str) -> list[tuple[str, list[str]]]:
    """Extract equations (including chained a = b = c) per clause when '=' appears."""
    out: list[tuple[str, list[str]]] = []
    for cl in _split_clauses(text):
        if "=" not in cl:
            continue
        raw_parts = cl.split("=")
        if len(raw_parts) < 2:
            continue
        lhs0 = _strip_leading_prose_prefix(raw_parts[0])
        if not lhs0:
            continue
        rhs_tail_raw = raw_parts[-1].strip()
        m_rhs = _EQUATION_RHS_EXPR_RE.match(rhs_tail_raw)
        rhs_last = (m_rhs.group(1) if m_rhs else rhs_tail_raw).rstrip(".,;: ")
        # Unwrap parenthetical equations like "... en 2026 (13 + (2026 - 1933) = 106)"
        while rhs_last.endswith(")") and rhs_last.count(")") > rhs_last.count("("):
            unclosed = _unclosed_open_paren_indices(lhs0)
            if not unclosed:
                break
            lhs0 = _strip_leading_prose_prefix(lhs0[unclosed[0] + 1 :].strip())
            rhs_last = rhs_last[:-1].rstrip(".,;: ")
        if not lhs0 or not rhs_last:
            continue
        sides = [lhs0] + [p.strip() for p in raw_parts[1:-1]] + [rhs_last]
        combined = " = ".join(sides)
        has_math = bool(re.search(r"[\d+\-*/×÷]", lhs0) and re.search(r"[\dA-Za-z+\-*/×÷]", rhs_last))
        if not has_math:
            continue
        out.append((combined, sides))
    return out


def verify_response_equations(
    text: str,
    graph: JustificationGraph,
    known_doc_nums: set[Decimal] | None = None,
    target_vals_to_match: list[Decimal] | None = None,
    *,
    doc_year_diffs: set[Decimal] | None = None,
) -> tuple[list[EquationCheckResult], list[dict[str, Any]]]:
    """Parse and deterministically verify inline equations against AST Decimal arithmetic & graph values."""
    if known_doc_nums is None:
        known_doc_nums = set()
        for node in graph.claims.values():
            dn = _to_dec(node.value)
            if dn is not None:
                known_doc_nums.add(dn)
        for step in graph.steps:
            for pv in (step.parameters or {}).values():
                dn = _to_dec(pv)
                if dn is not None:
                    known_doc_nums.add(dn)
    if target_vals_to_match is None:
        target_vals_to_match = []
        conc_node = graph.claims.get(graph.conclusion_id)
        if conc_node is not None:
            cn = _to_dec(conc_node.value)
            if cn is not None:
                target_vals_to_match.append(cn)

    checks: list[EquationCheckResult] = []
    issues: list[dict[str, Any]] = []
    conc_id = graph.conclusion_id
    step_ids = [s.step_id for s in graph.steps if s.conclusion_id == conc_id]
    primary_step_id = step_ids[0] if step_ids else None
    year_diffs = doc_year_diffs or set()

    for raw_eq, sides in _extract_equation_chains(text):
        lhs_expr = sides[0]
        rhs_expr = sides[-1]
        side_evals = [_eval_arithmetic_side(s) for s in sides]
        if any(val is None for val, _ in side_evals):
            checks.append(EquationCheckResult(
                raw_equation=raw_eq,
                lhs_expr=lhs_expr,
                rhs_expr=rhs_expr,
                lhs_value=side_evals[0][0],
                rhs_value=side_evals[-1][0],
                is_mathematically_true=None,
                matches_graph_step=False,
                matched_step_id=None,
                status="unknown",
                reason="UNVERIFIED_EXPRESSION_IN_RESPONSE",
            ))
            issues.append({
                "code": "UNVERIFIED_EXPRESSION_IN_RESPONSE",
                "status": "unknown",
                "reason": f"Mathematical expression '{raw_eq}' could not be deterministically parsed/evaluated.",
                "severity": "warning",
                "equation": raw_eq,
            })
            continue

        vals = [v for v, _ in side_evals if v is not None]
        all_leaves = [lf for _, lfs in side_evals for lf in lfs]
        first_val = vals[0]
        math_ok = all(abs(v - first_val) <= Decimal("0.01") for v in vals[1:])
        if not math_ok:
            checks.append(EquationCheckResult(
                raw_equation=raw_eq,
                lhs_expr=lhs_expr,
                rhs_expr=rhs_expr,
                lhs_value=vals[0],
                rhs_value=vals[-1],
                is_mathematically_true=False,
                matches_graph_step=False,
                matched_step_id=None,
                status="refuted",
                reason="FALSE_EQUATION_IN_RESPONSE",
            ))
            issues.append({
                "code": "FALSE_EQUATION_IN_RESPONSE",
                "status": "refuted",
                "reason": (
                    f"Mathematical equality '{raw_eq}' is false: LHS evaluates to {vals[0]} "
                    f"while RHS evaluates to {vals[-1]}."
                ),
                "severity": "error",
                "equation": raw_eq,
                "lhs_value": str(vals[0]),
                "rhs_value": str(vals[-1]),
            })
            continue

        # Mathematically true: check operands and result correspond to graph steps / evidence
        ungrounded_operands = [
            str(lf) for lf in all_leaves
            if not any(abs(lf - kn) <= Decimal("0.0001") for kn in known_doc_nums)
        ]
        result_matches_target_or_substep = (
            not target_vals_to_match
            or any(abs(vals[-1] - tv) <= Decimal("0.01") for tv in target_vals_to_match)
            or any(abs(vals[-1] - yd) <= Decimal("0.01") for yd in year_diffs)
        )
        if ungrounded_operands or not result_matches_target_or_substep:
            checks.append(EquationCheckResult(
                raw_equation=raw_eq,
                lhs_expr=lhs_expr,
                rhs_expr=rhs_expr,
                lhs_value=vals[0],
                rhs_value=vals[-1],
                is_mathematically_true=True,
                matches_graph_step=False,
                matched_step_id=None,
                status="refuted",
                reason="UNGROUNDED_EQUATION_IN_RESPONSE",
            ))
            issues.append({
                "code": "UNGROUNDED_EQUATION_IN_RESPONSE",
                "status": "refuted",
                "reason": (
                    f"Equation '{raw_eq}' is mathematically valid ({vals[-1]}) but does not match verified "
                    "graph derivation values/operands."
                ),
                "severity": "error",
                "equation": raw_eq,
                "evaluated_value": str(vals[-1]),
                "ungrounded_operands": ungrounded_operands,
            })
            continue

        checks.append(EquationCheckResult(
            raw_equation=raw_eq,
            lhs_expr=lhs_expr,
            rhs_expr=rhs_expr,
            lhs_value=vals[0],
            rhs_value=vals[-1],
            is_mathematically_true=True,
            matches_graph_step=True,
            matched_step_id=primary_step_id,
            status="verified",
            reason="EQUATION_VERIFIED",
        ))

    return checks, issues


# ---------------------------------------------------------------------------
# Structured condition & polarity comparator
# ---------------------------------------------------------------------------

def _condition_content_tokens(text_val: str) -> tuple[str, ...]:
    tokens = []
    for raw in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text_val):
        nt = _norm_token(raw)
        if len(nt) >= 3 and nt not in _CONDITION_SYNTAX_WORDS:
            tokens.append(nt)
    return tuple(dict.fromkeys(tokens))


def _required_condition_structure(cond: Mapping[str, Any] | Any) -> StructuredCondition:
    if isinstance(cond, Mapping):
        fld = str(cond.get("field") or "")
        val = str(cond.get("value") or "")
        pol_hint = str(cond.get("polarity") or "").lower()
        raw_text = f"{fld}: {val}".strip(": ") if fld else val
    else:
        fld = ""
        val = str(cond)
        pol_hint = ""
        raw_text = val

    low_val = val.lower()
    low_fld = fld.lower()
    if (
        pol_hint in ("excluding", "negative", "exception")
        or low_fld in ("exception", "excepcion", "excepción", "exclusion", "salvedad")
        or bool(_EXCEPTION_MARKER_RE.match(low_val.strip()))
    ):
        polarity = "excluding"
        kind = "exception"
    else:
        polarity = "requiring"
        kind = "prerequisite"

    # Skip purely generic field label words when building required content tokens
    raw_token_source = val if val and low_fld in ("exception", "excepcion", "excepción", "condition", "condicion", "condición", "proviso", "rule", "scope") else f"{fld} {val}"
    return StructuredCondition(
        raw_text=raw_text,
        kind=kind,
        effective_polarity=polarity,
        content_tokens=_condition_content_tokens(raw_token_source),
    )


def _classify_response_span_polarity(prefix: str) -> str:
    """Classify the logical condition operator governing tokens in a response segment."""
    # Narrow window starting at last clause/colon separator right before matched tokens when needed
    if _CONCESSIVE_OVERRIDE_RE.search(prefix):
        return "concessive_override"
    # Check exception markers closest to the condition terms (e.g. "Sujeto a ...: salvo en caso de ...")
    sub = prefix.split(":")[-1]
    if _EXCEPTION_MARKER_RE.search(sub) or _EXCEPTION_MARKER_RE.search(prefix):
        return "excluding"
    if _PREREQUISITE_MARKER_RE.search(sub) or _PREREQUISITE_MARKER_RE.search(prefix):
        return "requiring"
    return "unmarked"


def _evaluate_structured_conditions(
    text: str,
    clauses: list[str],
    req_conds: Sequence[Any],
    doc_and_q_lexicon: set[str],
    *,
    semantic_evaluator: Any = None,
) -> tuple[bool, bool, list[StructuredCondition], list[dict[str, Any]]]:
    """Compare required structured conditions & polarity against natural-language response."""
    if not req_conds:
        return False, False, [], []

    structured_reqs = [_required_condition_structure(c) for c in req_conds]
    # Deduplicate identical structured conditions across premise/conclusion nodes
    dedup_reqs: list[StructuredCondition] = []
    seen_req_keys: set[tuple[str, tuple[str, ...]]] = set()
    for sr in structured_reqs:
        key = (sr.effective_polarity, sr.content_tokens)
        if key not in seen_req_keys:
            seen_req_keys.add(key)
            dedup_reqs.append(sr)

    norm_clauses = [(_norm_token(cl), cl) for cl in clauses]
    # Also scan comma-delimited subclauses so conditions attached after ", incluso..." / ", salvo..." have exact local prefix
    cond_segments: list[tuple[str, str]] = []
    for cl in clauses:
        for seg in re.split(r",\s*", cl):
            seg_c = seg.strip()
            if seg_c:
                cond_segments.append((_norm_token(seg_c), seg_c))

    condition_dropped = False
    condition_polarity_inverted = False
    issues: list[dict[str, Any]] = []
    observed_structures: list[StructuredCondition] = []

    all_req_tokens = {tok for req in dedup_reqs for tok in req.content_tokens}

    for req in dedup_reqs:
        if not req.content_tokens:
            # Generic marker-only fallback if a condition carries no content tokens
            has_marker = bool(_CONDITIONAL_MARKER_RE.search(text))
            if not has_marker:
                condition_dropped = True
                issues.append({
                    "code": "CONDITION_DROPPED_IN_RESPONSE",
                    "status": "refuted",
                    "reason": f"Response omits required condition '{req.raw_text}'.",
                    "severity": "error",
                })
            continue

        # Find best matching segment (with highest overlap of required content tokens)
        best_seg: tuple[str, str] | None = None
        best_matched: list[str] = []
        for norm_seg, raw_seg in cond_segments:
            seg_toks = {_norm_token(w) for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", norm_seg)}
            matched = [t for t in req.content_tokens if t in seg_toks]
            if len(matched) > len(best_matched):
                best_matched = matched
                best_seg = (norm_seg, raw_seg)
        if not best_matched:
            for norm_cl, raw_cl in norm_clauses:
                cl_toks = {_norm_token(w) for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", norm_cl)}
                matched = [t for t in req.content_tokens if t in cl_toks]
                if len(matched) > len(best_matched):
                    best_matched = matched
                    best_seg = (norm_cl, raw_cl)

        coverage = len(best_matched) / max(1, len(req.content_tokens))

        if coverage == 0.0:
            condition_dropped = True
            issues.append({
                "code": "CONDITION_DROPPED_IN_RESPONSE",
                "status": "refuted",
                "reason": f"Response omits mandatory condition/exception '{req.raw_text}'.",
                "severity": "error",
                "required_condition": req.to_dict(),
            })
            # Check if response substituted an unrelated conditional clause (e.g. "si hace sol")
            for norm_seg, raw_seg in cond_segments:
                if _CONDITIONAL_MARKER_RE.search(raw_seg):
                    seg_toks = _condition_content_tokens(raw_seg)
                    ungrounded_cond_toks = [
                        t for t in seg_toks
                        if t not in all_req_tokens
                        and t not in doc_and_q_lexicon
                        and t not in _EPISTEMIC_META_TOKENS
                    ]
                    if ungrounded_cond_toks:
                        issues.append({
                            "code": "UNSUPPORTED_CONDITION_SUBSTITUTION",
                            "status": "refuted",
                            "reason": (
                                f"Response substituted an ungrounded condition '{raw_seg}' "
                                f"(tokens={ungrounded_cond_toks}) instead of required '{req.raw_text}'."
                            ),
                            "severity": "error",
                            "substituted_clause": raw_seg,
                            "ungrounded_tokens": ungrounded_cond_toks,
                        })
                        break
            continue

        assert best_seg is not None
        _, raw_match_seg = best_seg
        obs_pol = _classify_response_span_polarity(raw_match_seg)
        observed_structures.append(StructuredCondition(
            raw_text=raw_match_seg,
            kind=obs_pol if obs_pol != "requiring" else "prerequisite",
            effective_polarity=obs_pol,
            content_tokens=tuple(best_matched),
        ))

        if obs_pol == "concessive_override" or (
            obs_pol in ("excluding", "requiring") and obs_pol != req.effective_polarity
        ):
            condition_dropped = True
            condition_polarity_inverted = True
            issues.append({
                "code": "CONDITION_POLARITY_INVERTED",
                "status": "refuted",
                "reason": (
                    f"Condition '{req.raw_text}' requires '{req.effective_polarity}' semantics, "
                    f"but response expressed '{obs_pol}' in '{raw_match_seg}'."
                ),
                "severity": "error",
                "required_polarity": req.effective_polarity,
                "observed_polarity": obs_pol,
                "clause": raw_match_seg,
            })
        elif obs_pol == "unmarked" or coverage < 0.5:
            # Ambiguous / incomplete correspondence without explicit semantic resolution
            sem_status = "unknown"
            if semantic_evaluator is not None and getattr(semantic_evaluator, "enabled", False) and getattr(semantic_evaluator, "callable_evaluator", None) is not None:
                try:
                    sem_res = semantic_evaluator.callable_evaluator(
                        check="condition_correspondence",
                        required_condition=req.raw_text,
                        response_clause=raw_match_seg,
                    )
                    if isinstance(sem_res, Mapping) and sem_res.get("status") in ("verified", "refuted"):
                        sem_status = str(sem_res["status"])
                except Exception:
                    sem_status = "unknown"
            if sem_status == "verified":
                continue
            if sem_status == "refuted":
                condition_dropped = True
                issues.append({
                    "code": "CONDITION_DROPPED_IN_RESPONSE",
                    "status": "refuted",
                    "reason": f"Semantic check refuted equivalence between '{raw_match_seg}' and '{req.raw_text}'.",
                    "severity": "error",
                })
            else:
                issues.append({
                    "code": "CONDITION_CORRESPONDENCE_UNRESOLVED",
                    "status": "unknown",
                    "reason": (
                        f"Condition equivalence for '{req.raw_text}' could not be deterministically confirmed "
                        f"in '{raw_match_seg}' (polarity={obs_pol}, token_coverage={coverage:.2f})."
                    ),
                    "severity": "warning",
                    "required_condition": req.to_dict(),
                    "clause": raw_match_seg,
                })

    return condition_dropped, condition_polarity_inverted, observed_structures, issues


# ---------------------------------------------------------------------------
# Clause-level decomposition, grounding lexicon & conclusion polarity check
# ---------------------------------------------------------------------------

def _build_grounding_lexicon(
    graph: JustificationGraph,
    conc_id: str,
    documents: Mapping[str, SourceDocument] | None,
    q_text: str,
) -> tuple[set[str], set[str], set[str]]:
    """Build exact normalized tokens and conservative stems from documents, question, and graph."""
    exact_lex: set[str] = set()
    stem_lex: set[str] = set()
    doc_q_exact: set[str] = set()

    def _add_words(raw_str: str | None, *, from_doc_or_q: bool = False) -> None:
        if not raw_str:
            return
        for raw_tok in re.findall(r"[A-Za-zÀ-ÿ0-9]+", raw_str):
            nt = _norm_token(raw_tok)
            if len(nt) >= 3:
                exact_lex.add(nt)
                if len(nt) >= 6:
                    stem_lex.add(_stem_key(nt))
                if from_doc_or_q:
                    doc_q_exact.add(nt)

    for doc in (documents or {}).values():
        _add_words(doc.text, from_doc_or_q=True)
    if q_text:
        _add_words(q_text, from_doc_or_q=True)

    conc = graph.claims[conc_id]
    anc_ids = graph.ancestor_claims(conc_id)
    _PREDICATE_BILINGUAL_EQUIV: dict[str, tuple[str, ...]] = {
        "age": ("edad", "age"),
        "anchor_age": ("edad", "age"),
        "duration": ("duracion", "duration"),
        "duration_months": ("duracion", "duration", "meses", "months"),
        "amount": ("monto", "importe", "cantidad", "amount"),
        "total_amount": ("monto", "importe", "cantidad", "total", "amount"),
        "charge": ("cargo", "monto", "importe", "charge"),
        "refund": ("reembolso", "refund"),
        "approved_refund": ("reembolso", "aprobado", "approved", "refund"),
        "liability_cap": ("limite", "tope", "responsabilidad", "cap", "liability"),
        "revenue": ("ingreso", "ingresos", "revenue"),
        "combined_revenue": ("ingreso", "ingresos", "combinado", "revenue"),
    }
    for nid in [conc_id, *sorted(anc_ids)]:
        if nid not in graph.claims:
            continue
        node = graph.claims[nid]
        _add_words(node.subject)
        if node.predicate:
            pred_norm = node.predicate.lower().strip()
            _add_words(pred_norm.replace("_", " "))
            for eq_word in _PREDICATE_BILINGUAL_EQUIV.get(pred_norm, ()):
                _add_words(eq_word)
        _add_words(node.unit)
        if node.value is not None and isinstance(node.value, str):
            _add_words(node.value)
        for cond in node.conditions:
            if isinstance(cond, Mapping):
                _add_words(str(cond.get("field") or ""))
                _add_words(str(cond.get("value") or ""))
            else:
                _add_words(str(cond))
        for assump in node.assumptions:
            _add_words(str(assump).replace("_", " "))

    for syns in _UNIT_SYNONYMS.values():
        for syn in syns:
            _add_words(syn)

    # Also register exact stems for epistemic meta vocabulary
    meta_stems = {_stem_key(t) for t in _EPISTEMIC_META_TOKENS if len(t) >= 6}
    return exact_lex, stem_lex | meta_stems, doc_q_exact


def _is_graph_conclusion_negative(graph: JustificationGraph) -> bool:
    conc = graph.claims[graph.conclusion_id]
    if isinstance(conc.value, bool) and conc.value is False:
        return True
    if isinstance(conc.scope, Mapping) and str(conc.scope.get("polarity", "")).lower() == "negative":
        return True
    if conc.predicate and conc.predicate.lower().startswith("not_"):
        return True
    return False


def _decompose_and_verify_clauses(
    clauses: list[str],
    graph: JustificationGraph,
    target_vals_to_match: list[Decimal],
    exact_lex: set[str],
    stem_lex: set[str],
    subj: str | None,
    *,
    has_material_conflict: bool,
    semantic_evaluator: Any = None,
) -> tuple[bool, list[ResponseClauseClaim], list[dict[str, Any]]]:
    """Decompose text into clauses, check conclusion polarity inversion & ungrounded non-numeric claims."""
    conc_negative = _is_graph_conclusion_negative(graph)
    conc = graph.claims[graph.conclusion_id]
    clause_claims: list[ResponseClauseClaim] = []
    issues: list[dict[str, Any]] = []
    conclusion_polarity_inverted = False

    for cl in clauses:
        for sub_cl in _split_claim_subclauses(cl):
            cl_nums = tuple(_extract_clause_numbers(sub_cl))
            is_formula_or_disc = _clause_is_explanatory_formula_or_disclaimer(sub_cl)
            is_cond = bool(_CONDITIONAL_MARKER_RE.search(sub_cl)) and not is_formula_or_disc
            is_conflict = bool(_CONFLICT_DISCLOSURE_RE.search(sub_cl))
            mentions_target = bool(
                target_vals_to_match
                and any(any(abs(cn - tv) <= Decimal("0.0001") for tv in target_vals_to_match) for cn in cl_nums)
            )
            stripped_safe_disc = _SAFE_DISCLAIMER_NEGATION_RE.sub(" ", sub_cl)
            neg_hit = bool(_CONCLUSION_NEGATION_RE.search(stripped_safe_disc))
            polarity = "negative" if neg_hit else "affirmative"
            is_formula = "=" in sub_cl and any(op in sub_cl for op in ("+", "-", "*", "/", "×", "÷"))
            if is_formula:
                role = "formula"
            elif (mentions_target or _extract_ranges(sub_cl)) and neg_hit and not is_formula:
                role = "target_assertion"
            elif is_formula_or_disc:
                role = "disclaimer"
            elif mentions_target or _extract_ranges(sub_cl):
                role = "target_assertion"
            elif is_cond:
                role = "condition"
            elif is_conflict and has_material_conflict:
                role = "conflict_disclosure"
            else:
                role = "factual_assertion"

            if role == "target_assertion" and neg_hit and not conc_negative:
                conclusion_polarity_inverted = True
                issues.append({
                    "code": "CONCLUSION_POLARITY_INVERTED",
                    "status": "refuted",
                    "reason": (
                        f"Response negates or inverts polarity of the verified conclusion in '{sub_cl}'."
                    ),
                    "severity": "error",
                    "clause": sub_cl,
                })

            # Grounding check over substantive alphabetic content words (>= 3 chars)
            ungrounded: list[str] = []
            for raw_tok in re.findall(r"[A-Za-zÀ-ÿ]+", sub_cl):
                nt = _norm_token(raw_tok)
                if len(nt) < 3 or nt in _FUNCTION_WORDS or nt in _EPISTEMIC_META_TOKENS:
                    continue
                if nt in exact_lex:
                    continue
                if len(nt) >= 6 and _stem_key(nt) in stem_lex:
                    continue
                ungrounded.append(raw_tok)

            dedup_ungrounded = tuple(dict.fromkeys(ungrounded))
            is_grounded = len(dedup_ungrounded) == 0

            # Flag ungrounded substantive non-numeric assertions (avoid noise on pure formulas/evasion)
            if dedup_ungrounded and role != "formula":
                sem_verdict = "refuted"
                if (
                    semantic_evaluator is not None
                    and getattr(semantic_evaluator, "enabled", False)
                    and getattr(semantic_evaluator, "callable_evaluator", None) is not None
                ):
                    try:
                        sem_res = semantic_evaluator.callable_evaluator(
                            check="clause_grounding",
                            clause=sub_cl,
                            ungrounded_tokens=list(dedup_ungrounded),
                            conclusion=conc.to_dict(),
                        )
                        if isinstance(sem_res, Mapping) and sem_res.get("status") in ("verified", "unknown", "refuted"):
                            sem_verdict = str(sem_res["status"])
                    except Exception:
                        sem_verdict = "refuted"

                if sem_verdict == "verified":
                    is_grounded = True
                elif sem_verdict == "unknown":
                    issues.append({
                        "code": "RESPONSE_CORRESPONDENCE_PENDING_SEMANTIC",
                        "status": "unknown",
                        "reason": (
                            f"Clause '{sub_cl}' contains ungrounded terms {list(dedup_ungrounded)} "
                            "that require semantic entailment verification."
                        ),
                        "severity": "warning",
                        "clause": sub_cl,
                        "ungrounded_tokens": list(dedup_ungrounded),
                    })
                else:
                    issues.append({
                        "code": "UNSUPPORTED_FACTUAL_ADDITION",
                        "status": "refuted",
                        "reason": (
                            f"Clause '{sub_cl}' introduces ungrounded factual claim/attribute "
                            f"{list(dedup_ungrounded)} not supported by evidence or graph."
                        ),
                        "severity": "error",
                        "clause": sub_cl,
                        "ungrounded_tokens": list(dedup_ungrounded),
                    })

            clause_claims.append(ResponseClauseClaim(
                clause_text=sub_cl,
                role=role,
                subject=subj,
                polarity=polarity,
                numbers=cl_nums,
                is_grounded=is_grounded,
                ungrounded_tokens=dedup_ungrounded,
            ))

    return conclusion_polarity_inverted, clause_claims, issues


def verify_response_graph_correspondence(
    candidate_response: str | None,
    graph: JustificationGraph,
    *,
    question: str | Mapping[str, Any] | None = None,
    documents: Mapping[str, SourceDocument] | None = None,
    unit_registry: UnitRegistry | None = None,
    expected_subject: str | None = None,
    has_material_conflict: bool = False,
    semantic_evaluator: Any = None,
) -> dict[str, Any]:
    """Analyze claim-level, equational, conditional, and structural alignment between text and graph."""
    if candidate_response is None:
        return {
            "response_checked": False,
            "correspondence_ok": False,
            "issues": (),
        }

    text = _UNICODE_MINUS_RE.sub("-", candidate_response).strip()
    q_text = (
        _UNICODE_MINUS_RE.sub("-", question)
        if isinstance(question, str)
        else (
            _UNICODE_MINUS_RE.sub("-", str(question.get("text") or question.get("question") or ""))
            if isinstance(question, Mapping)
            else ""
        )
    )
    conc = graph.claims[graph.conclusion_id]
    anc_ids = graph.ancestor_claims(graph.conclusion_id)
    anc_nodes = [graph.claims[aid] for aid in sorted(anc_ids) if aid in graph.claims]
    clauses = _split_clauses(text)
    issues: list[dict[str, Any]] = []

    if not text:
        issues.append({
            "code": "EMPTY_CANDIDATE_RESPONSE",
            "status": "refuted",
            "reason": "Candidate response text is empty.",
            "severity": "error",
        })
        return {
            "response_checked": True,
            "correspondence_ok": False,
            "issues": tuple(issues),
        }

    # Gather expected target numbers / interval from conclusion
    conc_val = _to_dec(conc.value)
    conc_interval: tuple[Decimal, Decimal] | None = None
    if isinstance(conc.value, Mapping) and "min" in conc.value and "max" in conc.value:
        lo_d, hi_d = _to_dec(conc.value["min"]), _to_dec(conc.value["max"])
        if lo_d is not None and hi_d is not None:
            conc_interval = (lo_d, hi_d)

    all_nums_in_text = _extract_clause_numbers(text)
    ranges_in_text = _extract_ranges(text)

    # Known safe numbers present in documents, question, premise operands, or conclusion itself
    known_doc_nums: set[Decimal] = set()
    for doc in (documents or {}).values():
        known_doc_nums.update(_extract_clause_numbers(doc.text))
    if q_text:
        known_doc_nums.update(_extract_clause_numbers(q_text))
    for node in anc_nodes + [conc]:
        nv = _to_dec(node.value)
        if nv is not None:
            known_doc_nums.add(nv)
        elif isinstance(node.value, Mapping):
            for k in ("min", "max", "low", "high", "value"):
                d = _to_dec(node.value.get(k))
                if d is not None:
                    known_doc_nums.add(d)
    doc_years = {int(n) for n in known_doc_nums if 1500 <= n <= 2150 and n == int(n)}
    doc_year_diffs: set[Decimal] = set()
    for y1 in doc_years:
        for y2 in doc_years:
            if y1 > y2:
                d_diff = Decimal(y1 - y2)
                known_doc_nums.add(d_diff)
                doc_year_diffs.add(d_diff)
    for n in list(known_doc_nums):
        if abs(n) <= Decimal(1000) and n != 0:
            known_doc_nums.add(n / Decimal(100))

    target_vals_to_match: list[Decimal] = []
    if conc_val is not None:
        target_vals_to_match.append(conc_val)
    elif conc_interval is not None:
        target_vals_to_match.extend([conc_interval[0], conc_interval[1]])

    # 1. Precision / approximation consistency across ALL clauses
    approx_clauses: list[str] = []
    target_approx_clauses: list[str] = []
    exact_clauses: list[str] = []
    unqualified_target_clauses: list[str] = []

    for cl in clauses:
        has_approx = bool(_APPROX_MODIFIER_RE.search(cl) or _extract_ranges(cl))
        has_exact = _clause_asserts_exactness(cl)
        cl_nums = _extract_clause_numbers(cl)
        is_disclaimer_or_formula = _clause_is_explanatory_formula_or_disclaimer(cl)
        mentions_target_val = bool(
            target_vals_to_match
            and any(any(abs(cn - tv) <= Decimal("0.0001") for tv in target_vals_to_match) for cn in cl_nums)
        )
        mentions_target_unit = bool(
            conc.unit
            and any(syn in cl.lower() for syn in _UNIT_SYNONYMS.get(conc.unit, (conc.unit.lower(),)))
        )
        is_target_assertion_clause = (mentions_target_val or (ranges_in_text and _extract_ranges(cl))) and not is_disclaimer_or_formula

        if has_approx:
            approx_clauses.append(cl)
            if is_target_assertion_clause or mentions_target_unit:
                target_approx_clauses.append(cl)
        if has_exact:
            exact_clauses.append(cl)
        if is_target_assertion_clause and not has_approx:
            unqualified_target_clauses.append(cl)

    contradictory_precision = bool(approx_clauses and exact_clauses)
    if exact_clauses and any(_extract_ranges(c) for c in clauses):
        contradictory_precision = True

    if contradictory_precision:
        issues.append({
            "code": "CONTRADICTORY_PRECISION_QUALIFICATION",
            "status": "refuted",
            "reason": "Response asserts both an approximation/interval and an exact claim in the same answer.",
            "severity": "error",
            "clauses": {"approx": approx_clauses, "exact": exact_clauses},
        })

    response_approximated_or_ranged = bool(target_approx_clauses and not contradictory_precision and not exact_clauses)
    response_claims_exact_unqualified = bool(exact_clauses or (unqualified_target_clauses and not target_approx_clauses))

    # 2. Source attribution check
    has_source_attr = bool(_SOURCE_ATTRIBUTION_RE.search(text))
    has_contradictory_ext_truth = bool(_CONTRADICTORY_EXTERNAL_TRUTH_RE.search(text))
    source_qualified = bool(has_source_attr and not has_contradictory_ext_truth)
    if has_source_attr and has_contradictory_ext_truth:
        issues.append({
            "code": "CONTRADICTORY_SOURCE_ATTRIBUTION",
            "status": "refuted",
            "reason": "Response cites the supplied text while also claiming independently verified external truth.",
            "severity": "error",
        })

    # 3. Subject preservation check
    subj = expected_subject or conc.subject
    if not subj:
        for n in anc_nodes:
            if n.subject:
                subj = n.subject
                break
    subject_changed = False
    if subj:
        subj_tokens = [t for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", subj.lower()) if len(t) >= 3 and t not in _COMMON_STOPWORDS]
        if subj_tokens:
            low_text = text.lower()
            subj_in_text = any(tok in low_text for tok in subj_tokens)
            allowed_cap_words = set(subj_tokens)
            other_doc_entities: set[str] = set()
            unit_words_low = {s.lower() for syns in _UNIT_SYNONYMS.values() for s in syns}
            for doc in (documents or {}).values():
                doc_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", doc.text)
                allowed_cap_words.update(w.lower() for w in doc_words)
                for m_d in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})\b", doc.text):
                    w_low = m_d.group(1).lower()
                    if w_low not in _COMMON_STOPWORDS and w_low not in subj_tokens and w_low not in unit_words_low:
                        other_doc_entities.add(w_low)
            if q_text:
                allowed_cap_words.update(w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", q_text))
            allowed_cap_words.update(_COMMON_STOPWORDS)

            for m in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})*)\b", text):
                cand_words = [w.lower() for w in m.group(1).split()]
                prev_char = text[m.start() - 1] if m.start() > 0 else ""
                prev_two = text[max(0, m.start() - 2):m.start()].strip()
                at_sentence_start = m.start() == 0 or prev_char == "\n" or prev_two in {".", "!", "?", ":"}
                if len(cand_words) >= 2 and any(w not in allowed_cap_words for w in cand_words):
                    subject_changed = True
                    break
                if len(cand_words) == 1 and not at_sentence_start:
                    if cand_words[0] not in allowed_cap_words:
                        subject_changed = True
                        break
            if not subj_in_text and other_doc_entities:
                for other_ent in other_doc_entities:
                    if re.search(rf"\b{re.escape(other_ent)}\b", low_text):
                        subject_changed = True
                        break

            if subject_changed:
                issues.append({
                    "code": "SUBJECT_CHANGED_IN_REPAIR",
                    "status": "refuted",
                    "reason": f"Response altered subject '{subj}' or introduced an ungrounded entity.",
                    "severity": "error",
                })

    # 4. Calculation / magnitude fidelity, equation AST verification & evasion detection
    equation_checks, equation_issues = verify_response_equations(
        text,
        graph,
        known_doc_nums,
        target_vals_to_match,
        doc_year_diffs=doc_year_diffs,
    )
    issues.extend(equation_issues)

    evaded = bool(_EVASION_NO_ANSWER_RE.search(text))
    calc_mismatch = False
    if evaded and not has_material_conflict:
        issues.append({
            "code": "QUESTION_LEFT_UNANSWERED_EVASION",
            "status": "refuted",
            "reason": "Response omits or refuses to answer the core questioned claim instead of qualifying limitations.",
            "severity": "error",
        })
    elif target_vals_to_match and not evaded:
        asserted_nums: list[Decimal] = []
        for cl in clauses:
            if not _clause_is_explanatory_formula_or_disclaimer(cl):
                asserted_nums.extend(_extract_clause_numbers(cl))
        if not asserted_nums:
            asserted_nums = all_nums_in_text

        matched_graph_val = False
        for tv in target_vals_to_match:
            if any(abs(an - tv) <= Decimal("0.01") for an in asserted_nums):
                matched_graph_val = True
                break
        if conc_interval is not None and ranges_in_text:
            for r_lo, r_hi in ranges_in_text:
                if abs(r_lo - conc_interval[0]) <= Decimal("0.01") and abs(r_hi - conc_interval[1]) <= Decimal("0.01"):
                    matched_graph_val = True
                    break

        if not matched_graph_val:
            if asserted_nums:
                calc_mismatch = True
                issues.append({
                    "code": "CALCULATION_ALTERED_OR_INCONSISTENT",
                    "status": "refuted",
                    "reason": f"Response numbers {[str(x) for x in asserted_nums]} do not match verified conclusion value {conc.value!r}.",
                    "severity": "error",
                })
            else:
                issues.append({
                    "code": "QUESTION_LEFT_UNANSWERED_EVASION",
                    "status": "refuted",
                    "reason": "Response removed the concluding value without addressing the question.",
                    "severity": "error",
                })

    # 5. Unit preservation check
    if conc.unit:
        low_text = text.lower()
        target_unit_syns = _UNIT_SYNONYMS.get(conc.unit, (conc.unit.lower(),))

        def _unit_tok_re(tok: str) -> str:
            return rf"(?<!\w){re.escape(tok)}(?!\w)" if tok.isalpha() else re.escape(tok)

        target_unit_present = any(re.search(_unit_tok_re(s), low_text) for s in target_unit_syns)
        wrong_units_detected: list[str] = []
        for uname, syns in _UNIT_SYNONYMS.items():
            if uname == conc.unit:
                continue
            if unit_registry is not None:
                try:
                    u_c = unit_registry.lookup(conc.unit)
                    u_o = unit_registry.lookup(uname)
                    if u_c.dimension == u_o.dimension:
                        continue
                except Exception:
                    pass
            bound_to_number = any(
                re.search(
                    rf"(?:\d+(?:[.,]\d+)?\s*(?:(?:a|y|-|to)\s*\d+(?:[.,]\d+)?\s*)?{_unit_tok_re(s)}|{_unit_tok_re(s)}\s*\d+)",
                    low_text,
                )
                for s in syns
            )
            bare_present = any(re.search(_unit_tok_re(s), low_text) for s in syns)
            in_resp = bound_to_number or (bare_present and not target_unit_present)
            in_sources = any(
                any(s in d.text.lower() for s in syns) for d in (documents or {}).values()
            ) or (q_text and any(s in q_text.lower() for s in syns))
            if in_resp and not in_sources:
                wrong_units_detected.append(uname)
        if wrong_units_detected:
            issues.append({
                "code": "UNIT_ALTERED_IN_REPAIR",
                "status": "refuted",
                "reason": f"Response uses unexpected unit(s) {wrong_units_detected} instead of '{conc.unit}'.",
                "severity": "error",
            })

    # Build normalized grounding lexicon before structured condition & clause checks
    exact_lex, stem_lex, doc_q_exact = _build_grounding_lexicon(graph, graph.conclusion_id, documents, q_text)

    # 6. Structured condition & polarity comparison
    req_conds = list(conc.conditions)
    for n in anc_nodes:
        req_conds.extend(n.conditions)
    condition_dropped, condition_polarity_inverted, observed_conds, cond_issues = _evaluate_structured_conditions(
        text,
        clauses,
        req_conds,
        doc_q_exact,
        semantic_evaluator=semantic_evaluator,
    )
    issues.extend(cond_issues)

    # 7. Subtotal vs unqualified total check
    qualified_as_subtotal = bool(_SUBTOTAL_QUALIFICATION_RE.search(text))
    unqualified_total_claim = False
    for cl in clauses:
        if _UNQUALIFIED_TOTAL_RE.search(cl) and not _NEGATED_LIMIT_CONTEXT_RE.search(cl):
            unqualified_total_claim = True
            break
    presents_subtotal_as_total = bool(unqualified_total_claim and not qualified_as_subtotal)

    # 8. Material conflict disclosure check
    conflict_disclosed = bool(_CONFLICT_DISCLOSURE_RE.search(text))
    if has_material_conflict and not conflict_disclosed:
        issues.append({
            "code": "CONFLICT_HIDDEN_IN_RESPONSE",
            "status": "refuted",
            "reason": "Response fails to disclose unresolved contradicting evidence.",
            "severity": "error",
        })

    # 9. Claim-by-claim decomposition: conclusion polarity + non-numeric & numeric additions
    conclusion_polarity_inverted, clause_claims, clause_issues = _decompose_and_verify_clauses(
        clauses,
        graph,
        target_vals_to_match,
        exact_lex,
        stem_lex,
        subj,
        has_material_conflict=has_material_conflict,
        semantic_evaluator=semantic_evaluator,
    )
    existing_issue_keys = {(i.get("code"), i.get("clause")) for i in issues}
    for ci in clause_issues:
        # Avoid duplicate UNSUPPORTED_FACTUAL_ADDITION when UNSUPPORTED_CONDITION_SUBSTITUTION already named that segment
        if (
            ci.get("code") == "UNSUPPORTED_FACTUAL_ADDITION"
            and any(
                i.get("code") == "UNSUPPORTED_CONDITION_SUBSTITUTION" and i.get("substituted_clause") in ci.get("clause", "")
                for i in issues
            )
        ):
            continue
        k = (ci.get("code"), ci.get("clause"))
        if k not in existing_issue_keys:
            existing_issue_keys.add(k)
            issues.append(ci)

    doc_and_q_text = " ".join([d.text for d in (documents or {}).values()] + ([q_text] if q_text else []))
    doc_dates = set(_YEAR_OR_DATE_TOKEN_RE.findall(doc_and_q_text))
    resp_dates = set(_YEAR_OR_DATE_TOKEN_RE.findall(text))
    invented_dates = sorted(
        d for d in resp_dates
        if d not in doc_dates and _to_dec(d) != conc_val
    )
    if invented_dates:
        issues.append({
            "code": "UNSUPPORTED_FACTUAL_ADDITION",
            "status": "refuted",
            "reason": f"Response introduces ungrounded dates/years not present in evidence: {invented_dates}.",
            "severity": "error",
        })

    unexpected_nums = [
        str(n) for n in all_nums_in_text
        if not any(abs(n - kn) <= Decimal("0.0001") for kn in known_doc_nums)
        and not (ranges_in_text and any(lo <= n <= hi for lo, hi in ranges_in_text))
    ]
    if unexpected_nums and not calc_mismatch and not any(i.get("code") == "FALSE_EQUATION_IN_RESPONSE" for i in issues):
        issues.append({
            "code": "UNSUPPORTED_FACTUAL_ADDITION",
            "status": "refuted",
            "reason": f"Response introduces ungrounded numeric values not derivable from evidence: {unexpected_nums}.",
            "severity": "error",
        })

    correspondence_ok = len(issues) == 0
    return {
        "response_checked": True,
        "correspondence_ok": correspondence_ok,
        "response_approximated_or_ranged": response_approximated_or_ranged,
        "response_claims_exact_unqualified": response_claims_exact_unqualified,
        "contradictory_precision_qualification": contradictory_precision,
        "source_qualified": source_qualified,
        "contradictory_source_attribution": has_source_attr and has_contradictory_ext_truth,
        "condition_dropped_in_response": condition_dropped,
        "condition_polarity_inverted": condition_polarity_inverted,
        "conclusion_polarity_inverted": conclusion_polarity_inverted,
        "qualified_as_subtotal": qualified_as_subtotal,
        "presents_subtotal_as_total": presents_subtotal_as_total,
        "conflict_disclosed": conflict_disclosed,
        "has_missing_precision_disclosure": bool(_MISSING_PRECISION_DISCLOSURE_RE.search(text)),
        "equation_checks": [ec.to_dict() for ec in equation_checks],
        "structured_conditions": [sc.to_dict() for sc in observed_conds],
        "clause_claims": [cc.to_dict() for cc in clause_claims],
        "issues": tuple(issues),
    }
