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
from .corpus_obligations import _value_in_segment_with_unit
from .justification import JustificationGraph
from .vocabulary import UnitRegistry, label_key


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
class GraphRelationTuple:
    """Normalized structured relation supported by the JustificationGraph and evidence."""

    claim_id: str
    subject: str | None
    predicate: str | None
    slot_family: str  # "calendar_date" | "duration_or_age" | "monetary_amount" | "percentage_or_rate" | "quantity"
    value: Decimal | None
    interval: tuple[Decimal, Decimal] | None
    unit: str | None
    temporal_anchor: int | None
    source_entity: str | None = None
    event_tokens: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "slot_family": self.slot_family,
            "value": str(self.value) if self.value is not None else None,
            "interval": [str(self.interval[0]), str(self.interval[1])] if self.interval is not None else None,
            "unit": self.unit,
            "temporal_anchor": self.temporal_anchor,
            "source_entity": self.source_entity,
            "event_tokens": list(self.event_tokens),
        }


@dataclass(frozen=True)
class ResponseRelationTuple:
    """Structured numeric/entity relation extracted from a response sub-clause."""

    clause_text: str
    subject: str | None
    slot_family: str
    value: Decimal | None
    unit: str | None
    temporal_anchor: int | None
    status: str  # "verified" | "refuted" | "unknown"
    matched_claim_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_text": self.clause_text,
            "subject": self.subject,
            "slot_family": self.slot_family,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "temporal_anchor": self.temporal_anchor,
            "status": self.status,
            "matched_claim_id": self.matched_claim_id,
            "reason": self.reason,
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
    relations: tuple[ResponseRelationTuple, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_text": self.clause_text,
            "role": self.role,
            "subject": self.subject,
            "polarity": self.polarity,
            "numbers": [str(n) for n in self.numbers],
            "is_grounded": self.is_grounded,
            "ungrounded_tokens": list(self.ungrounded_tokens),
            "relations": [r.to_dict() for r in self.relations],
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
    r"(?:(?:el|la|los|las|este|esta|dicho|dicha|lo|the|supplied|provided|given|present|"
    r"master|governing|executed|draft|primary|secondary|underlying)\s+)*"
    r"(?:textos?|pasajes?|documentos?|extractos?|fragmentos?|fuentes?|evidencias?|"
    r"facturas?|datos|informaci[oó]n|relatos?|aportad[oa]s?|facilitad[oa]s?|"
    r"proporcionad[oa]s?|suministrad[oa]s?|passages?|texts?|documents?|excerpts?|sources?|evidence|"
    r"indenture|memorandum|offering\s+memorandum|agreement|contract|deed|filing|section|record|"
    r"draft|email|schedule|escritura|contrato|memorando|anexo|secci[oó]n)"
)
_SOURCE_ATTRIBUTION_RE = re.compile(
    rf"\b(?:{_ATTRIB_PREP}\s+{_ATTRIB_SOURCE_NOUN}|"
    r"in\s+(?:the\s+|draft\s+)?(?:indenture|offering\s+memorandum|om|memorandum|associate\s+email|email|section|excerpt|document|source)|"
    r"(?:indenture|offering\s+memorandum|om|document|source)\s+(?:states?|indicates?|provides?|governs?)|"
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
    r"divergent|inconclusive|governs?\s+over|whereas\b[^.;\n]{1,120}\b(?:states?|governs?)|"
    r"mientras\s+que\b[^.;\n]{1,120}\b(?:indica|establece|prevalece))\b",
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

_NUMBER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(-?(?:[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?|[1-9]\d{0,2}(?:\.\d{3})+,\d+|\d+(?:[.,]\d+)?))(?![A-Za-z0-9_.-])"
)
_YEAR_OR_DATE_TOKEN_RE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?|(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\b")

_UNIT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "years": ("año", "años", "year", "years", "yr", "yrs"),
    "months": ("mes", "meses", "month", "months"),
    "calendar_days": ("día", "días", "day", "days", "calendar days", "calendar day", "días naturales"),
    "business_days": ("días hábiles", "días laborables", "business day", "business days", "working days", "working day"),
    "USD": ("usd", "dólar", "dólares", "dollar", "dollars", "$"),
    "USD_per_year": ("usd/year", "usd per year", "usd/yr", "usd annually", "usd per annum", "dollars per year", "dólares anuales", "usd anuales", "usd", "$"),
    "EUR": ("eur", "euro", "euros", "€"),
    "GBP": ("gbp", "libra", "libras", "pound", "pounds", "£"),
    "percent": ("%", "por ciento", "percent", "pct", "per cent"),
    "percent_per_annum": ("%", "% per annum", "% p.a.", "percent per annum", "percent per year", "per annum", "anual"),
    "percentage_point": ("percentage point", "percentage points", "pp", "punto porcentual", "puntos porcentuales"),
    "percentage_points_per_annum": ("percentage points per annum", "pp per annum", "percentage points", "pp", "bps"),
    "basis_point": ("basis point", "basis points", "bp", "bps", "punto básico", "puntos básicos", "punto basico", "puntos basicos"),
    "basis_points_per_annum": ("basis points per annum", "bps per annum", "bps/yr", "bp/yr", "bps", "basis points"),
    "hours": ("hora", "horas", "hour", "hours", "hr", "hrs"),
    "minutes": ("minuto", "minutos", "minute", "minutes", "min"),
}

_FINANCIAL_ASSUMPTION_DISCLOSURE_RE = re.compile(
    r"\b(?:constant\s+principal|fixed\s+principal|unchanged\s+principal|constant\s+notional|"
    r"simple\s+annual(?:ized)?|illustrative\s+(?:annual|interest)|annualized\s+impact|"
    r"no\s+amortization|without\s+amortization|day[- ]count|act/360|30/360|"
    r"assuming\s+(?:constant|fixed|no\s+amortization)|"
    r"capital\s+(?:constante|fijo)|principal\s+constante|c[aá]lculo\s+anual\s+simple|"
    r"impacto\s+anual\s+ilustrativo|ilustrativ[oa]|sin\s+amortizaci[oó]n|"
    r"bajo\s+el\s+supuesto)\b",
    re.IGNORECASE,
)

_UNQUALIFIED_DAMAGES_CLAIM_RE = re.compile(
    r"\b(?:exact\s+legal\s+damages|guaranteed\s+(?:realized\s+)?(?:cashflow\s+)?loss|"
    r"automatic\s+(?:legal\s+)?damages|actual\s+realized\s+damages\s+of|"
    r"da[ñn]o(?:s)?\s+legal(?:es)?\s+exacto(?:s)?|p[eé]rdida\s+realizada\s+garantizada|"
    r"indemnizaci[oó]n\s+autom[aá]tica)\b",
    re.IGNORECASE,
)

_RELATIVE_PCT_CONFUSION_RE = re.compile(
    r"\b(?:relative\s+(?:percentage|percent)\s+(?:change|difference|increase|decrease)|"
    r"porcentaje\s+relativo|variaci[oó]n\s+porcentual\s+relativa)\b",
    re.IGNORECASE,
)

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
    "govern", "governs", "governing", "rank", "executed", "states", "versus", "correcting",
    "erroneously", "understatement", "overstatement", "comprehensive", "report", "puts",
    "non", "existent", "part", "over",
    "total", "sum", "items", "listed", "included", "population", "condition", "exception",
    "subject", "conditional", "contingent", "conflict", "contradiction", "discrepancy",
    "divergent", "unresolved", "inconclusive", "combined", "answer", "question", "today", "now", "currently",
    "assumption", "assumptions", "constant", "unchanged", "simple", "annual", "annualized", "illustrative",
    "amortization", "amortisation", "spread", "basis", "point", "points", "percentage", "percent",
    "assuming", "supuesto", "supuestos", "constante", "anual", "anualizado", "ilustrativo", "ilustrativa",
    "amortizacion", "diferencial", "puntos", "porcentuales", "basicos",
})

_FINITE_CLAIM_VERB_ALTS = (
    r"es|son|(?!era\s+\d{4})era|fueron|fue|est[aá]|est[aá]n|tiene|tienen|ten[ií]a|tendr[ií]a|"
    r"trabaja|vive|naci[oó]|muri[oó]|falleci[oó]|posee|ostenta|act[uú]a|participa|dirige|"
    r"compr[oó]|obtuvo|inici[oó]|firm[oó]|report[oó]|indica|indican|se[ñn]ala|asciende|alcanza|"
    r"corresponde|aplica|equivale|suman|"
    r"is|are|was|were|has|have|had|works|lives|born|died|owns|leads|bought|indicates|equals|totals"
)

_CONJ_VERB_SPLIT_RE = re.compile(
    rf"\s+(?:y|e|and)\s+(?=(?:{_FINITE_CLAIM_VERB_ALTS})\b)",
    re.IGNORECASE,
)

_COORD_SUBJ_VERB_SPLIT_RE = re.compile(
    rf"(?:\s+(?:y|e|and|pero|sino|mientras\s+que|whereas|while)\s+|,\s+)"
    rf"(?=(?:(?:el|la|los|las|un|una|the|this|that)\s+)?[A-Za-zÀ-ÿ0-9_-]{{2,}}"
    rf"(?:\s+[A-Za-zÀ-ÿ0-9_-]{{1,20}}){{0,3}}\s+(?:{_FINITE_CLAIM_VERB_ALTS})\b)",
    re.IGNORECASE,
)

_LEFT_HAS_VERB_OR_NUM_RE = re.compile(
    rf"(?:\b(?:{_FINITE_CLAIM_VERB_ALTS}|era)\b|\d)",
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


_US_THOUSANDS_RE = re.compile(r"^-?[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?$")
_EU_THOUSANDS_COMMA_RE = re.compile(r"^-?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?$")


def _normalize_numeric_str(raw_s: str) -> str:
    s = _UNICODE_MINUS_RE.sub("-", raw_s.strip()).replace(" ", "")
    if _US_THOUSANDS_RE.match(s):
        return s.replace(",", "")
    if _EU_THOUSANDS_COMMA_RE.match(s):
        return s.replace(".", "").replace(",", ".")
    return s.replace(",", ".")


def _to_dec(raw: Any) -> Decimal | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, Decimal):
        return raw if raw.is_finite() else None
    try:
        s = _normalize_numeric_str(str(raw))
        d = Decimal(s)
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _split_clauses(text: str) -> list[str]:
    norm = _UNICODE_MINUS_RE.sub("-", text or "")
    parts = re.split(
        r"(?:(?:(?<!\d)\.(?!\d)|[;!?\n]+)+|,\s*(?=(?:por tanto|por lo tanto|as[ií] que|pero|sin embargo|no obstante|therefore|hence|thus|so|however|but)\b))",
        norm,
    )
    return [p.strip() for p in parts if p and p.strip()]


def _split_coord_subject_verb(segment: str) -> list[str]:
    pieces: list[str] = []
    cur_start = 0
    for m in _COORD_SUBJ_VERB_SPLIT_RE.finditer(segment):
        left = segment[cur_start : m.start()].strip(" ,;:")
        right = segment[m.end() :].strip(" ,;:")
        if not left or not right:
            continue
        # Never split conjoined subjects before a single shared verb ("Factura A y Factura B suman 1000 USD")
        if not _LEFT_HAS_VERB_OR_NUM_RE.search(left):
            continue
        # Never split inside numeric ranges ("entre 60 y 84")
        if re.search(r"\b(?:entre|between|de|from)\s+\d+(?:[.,]\d+)?$", left, re.IGNORECASE):
            continue
        # Preserve subordinate "Cuando <anchor>, <main event>" when right adds no numeric value
        is_comma = segment[m.start() : m.end()].lstrip().startswith(",")
        if is_comma and re.match(r"^(?:cuando|cuando\s+el|if|when|since|dado\s+que)\b", left, re.IGNORECASE):
            if not _NUMBER_TOKEN_RE.search(right):
                continue
        pieces.append(left)
        cur_start = m.end()
    tail = segment[cur_start:].strip(" ,;:")
    if tail:
        pieces.append(tail)
    return pieces


def _split_claim_subclauses(clause: str) -> list[str]:
    out: list[str] = []
    for disc_piece in _DISCOURSE_SPLIT_RE.split(clause):
        for verb_piece in _CONJ_VERB_SPLIT_RE.split(disc_piece):
            for coord_piece in _split_coord_subject_verb(verb_piece):
                cleaned = coord_piece.strip(" ,;:")
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
    norm = _strip_accents(low)
    if not _NUMBER_TOKEN_RE.search(clause) and (
        _SAFE_DISCLAIMER_NEGATION_RE.search(low)
        or "depende unicamente del documento" in norm
        or "no ha sido verificada" in norm
    ):
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
    {s for syns in _UNIT_SYNONYMS.values() for s in syns if s not in {"$"} and any(ch.isalpha() for ch in s)},
    key=len,
    reverse=True,
)
_EQUATION_UNIT_PAT = "|".join(re.escape(u).replace(r"\ ", r"\s+") for u in _UNIT_WORDS_FOR_EQUATION)
_EQUATION_UNIT_RE = re.compile(
    rf"(?:(?<!\w)(?:{_EQUATION_UNIT_PAT})(?!\w)|[$€£%])",
    re.IGNORECASE,
)
_EQUATION_PROSE_LEAD_SPLIT_RE = re.compile(
    r"(?:\b(?:es|son|da|resulta|equivale\s+a|ser[ií]a|is|equals|gives|como|where|donde)\b|[:;]|,(?!\d))",
    re.IGNORECASE,
)
_EQUATION_RHS_EXPR_RE = re.compile(
    rf"^\s*((?:[-+*/×÷()\d.,\s$€£%]|\b(?:{_EQUATION_UNIT_PAT})\b)+)"
)
_RATE_PCT_SCALE_RE = re.compile(
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?)\s*"
    r"(?:%|(?<!\w)(?:pct|per\s+cent|por\s+ciento|pp|percentage\s+points?(?:\s+per\s+annum)?|puntos?\s+porcentuales?)(?!\w))",
    re.IGNORECASE,
)
_RATE_BPS_SCALE_RE = re.compile(
    r"(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?)\s*"
    r"(?<!\w)(?:bps?|basis\s+points?(?:\s+per\s+annum)?|puntos?\s+b[aá]sicos?)(?!\w)",
    re.IGNORECASE,
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
    # Pre-scale explicit percentage / pp / bps quantities so mixed-unit equations evaluate accurately
    cleaned = _RATE_BPS_SCALE_RE.sub(
        lambda m: f" ({_normalize_numeric_str(m.group(1))} * 0.0001) "
        if _to_dec(m.group(1)) is None
        else f" {str(_to_dec(m.group(1)) * Decimal('0.0001')).rstrip('0').rstrip('.') if '.' in str(_to_dec(m.group(1)) * Decimal('0.0001')) else str(_to_dec(m.group(1)) * Decimal('0.0001'))} ",
        cleaned,
    )
    cleaned = _RATE_PCT_SCALE_RE.sub(
        lambda m: f" ({_normalize_numeric_str(m.group(1))} * 0.01) "
        if _to_dec(m.group(1)) is None
        else f" {str(_to_dec(m.group(1)) * Decimal('0.01')).rstrip('0').rstrip('.') if '.' in str(_to_dec(m.group(1)) * Decimal('0.01')) else str(_to_dec(m.group(1)) * Decimal('0.01'))} ",
        cleaned,
    )
    cleaned = _EQUATION_UNIT_RE.sub(" ", cleaned)
    # Normalize remaining US/EU formatted numbers inside arithmetic string
    cleaned = _NUMBER_TOKEN_RE.sub(lambda m: _normalize_numeric_str(m.group(1)), cleaned)
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
        max_mag = max(abs(v) for v in vals)
        eq_tol = Decimal("0.01") if max_mag >= Decimal("1") else Decimal("1e-5")
        math_ok = all(abs(v - first_val) <= eq_tol for v in vals[1:])
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
        expanded_known = set(known_doc_nums)
        for kn in list(known_doc_nums):
            if abs(kn) <= Decimal(10000) and kn != 0:
                expanded_known.add(kn / Decimal(100))
                expanded_known.add(kn / Decimal(10000))
        ungrounded_operands = [
            str(lf) for lf in all_leaves
            if not any(abs(lf - kn) <= Decimal("0.00001") for kn in expanded_known)
        ]
        target_pool = list(target_vals_to_match)
        for tv in list(target_vals_to_match):
            if abs(tv) <= Decimal(10000) and tv != 0:
                target_pool.append(tv / Decimal(100))
                target_pool.append(tv / Decimal(10000))
        result_matches_target_or_substep = (
            not target_pool
            or any(abs(vals[-1] - tv) <= eq_tol for tv in target_pool)
            or any(abs(vals[-1] - yd) <= eq_tol for yd in year_diffs)
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
        _add_words(getattr(doc, "document_id", None), from_doc_or_q=True)
        rank = getattr(doc, "authority_rank", None)
        if rank is not None:
            _add_words(str(rank), from_doc_or_q=True)
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
    for nid in list(dict.fromkeys([conc_id, *sorted(anc_ids), *sorted(graph.claims.keys())])):
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


_DATE_PREDICATE_NAMES = frozenset({
    "anchor_date", "anchor_year", "reference_date", "reference_year",
    "start_date", "end_date", "birth_date", "date", "year",
})
_AGE_DURATION_PREDICATE_NAMES = frozenset({
    "age", "anchor_age", "duration", "duration_months", "notice_period", "sla_days",
})
_MONETARY_PREDICATE_NAMES = frozenset({
    "charge", "amount", "total_amount", "subtotal", "refund", "approved_refund",
    "liability_cap", "revenue", "combined_revenue", "base_revenue", "extra_revenue",
    "price", "cost", "fee", "payout",
})
_TEMPORAL_UNITS = frozenset({"years", "months", "days", "calendar_days", "business_days", "hours", "minutes"})
_MONETARY_UNITS = frozenset({"USD", "USD_per_year", "EUR", "GBP"})
_RATE_UNITS = frozenset({
    "percent", "percent_per_annum", "%", "percentage_point", "percentage_points_per_annum",
    "basis_point", "basis_points_per_annum", "bps",
})

_NON_COUNTED_POST_NUM_WORDS = frozenset({
    "indica", "indican", "senala", "senalan", "muestra", "muestran", "aplica", "aplicable",
    "equivale", "corresponde", "representa", "supera", "excede", "aproximados", "aproximadas",
    "exactos", "exactas", "totales", "parciales", "netos", "brutos", "vigente", "vigentes",
    "disponible", "disponibles", "respectivamente", "respectively", "roughly", "approximately",
    "indicates", "shows", "applies", "applicable", "equals", "represents", "exceeds",
})

_DATE_PRE_CONTEXT_RE = re.compile(
    r"(?:\b(?:en(?:\s+el\s+a[ñn]o)?|del\s+a[ñn]o|el\s+a[ñn]o|a[ñn]o|"
    r"para(?:\s+el\s+a[ñn]o)?|hacia|desde|hasta|durante|era|fue(?:\s+en)?|"
    r"fecha|nacido\s+en|naci[oó]\s+en|"
    r"in(?:\s+the\s+year)?|by|during|since|from|year|born\s+in)\s+)$",
    re.IGNORECASE,
)

_COPULA_OR_PRED_VERB_RE = re.compile(
    r"\b(?:es|son|era|eran|fue|fueron|ser[ií]a|tiene|tienen|ten[ií]a|tendr[ií]a|"
    r"indica|se[ñn]ala|asciende|alcanza|equivale|corresponde|aplica|suma|suman|"
    r"is|are|was|were|would\s+be|has|have|had|indicates|equals|totals)\b",
    re.IGNORECASE,
)

_AGGREGATE_PHRASE_RE = re.compile(
    r"\b(?:total|subtotal|suma|suman|combinad[oa]s?|conjunto|partidas|elementos|"
    r"entre\s+amb[oa]s|both|combined|sum|totals)\b",
    re.IGNORECASE,
)

_SOURCE_LEAD_ENTITY_RE = re.compile(
    r"^\s*([A-ZÁÉÍÓÚÑ][A-Za-zÀ-ÿ0-9_-]*(?:\s+[A-ZÁÉÍÓÚÑ0-9][A-Za-zÀ-ÿ0-9_-]*){0,2})\s+"
    r"(?:indica|se[ñn]ala|report[oóa]|establece|menciona|indicates|states|reports)\b"
)

_UNIT_WORD_MAP: dict[str, tuple[str, str]] = {}
for _u_canon, _u_syns in _UNIT_SYNONYMS.items():
    if _u_canon in _TEMPORAL_UNITS:
        _u_slot = "duration_or_age"
    elif _u_canon in _MONETARY_UNITS:
        _u_slot = "monetary_amount"
    elif _u_canon in _RATE_UNITS:
        _u_slot = "percentage_or_rate"
    else:
        _u_slot = "quantity"
    for _u_syn in _u_syns:
        _UNIT_WORD_MAP[_norm_token(_u_syn)] = (_u_canon, _u_slot)
_UNIT_WORD_MAP["$"] = ("USD", "monetary_amount")
_UNIT_WORD_MAP["€"] = ("EUR", "monetary_amount")
_UNIT_WORD_MAP["£"] = ("GBP", "monetary_amount")
_UNIT_WORD_MAP["%"] = ("percent", "percentage_or_rate")
_UNIT_WORD_MAP["año"] = ("years", "duration_or_age")
_UNIT_WORD_MAP["años"] = ("years", "duration_or_age")
_UNIT_WORD_MAP["dia"] = ("calendar_days", "duration_or_age")
_UNIT_WORD_MAP["dias"] = ("calendar_days", "duration_or_age")
_UNIT_WORD_MAP["dolar"] = ("USD", "monetary_amount")
_UNIT_WORD_MAP["dolares"] = ("USD", "monetary_amount")
_UNIT_WORD_MAP["days"] = ("calendar_days", "duration_or_age")


def _norm_phrase(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = re.sub(r"\s+", " ", _strip_accents(raw.lower())).strip()
    return cleaned or None


def _extract_year_int(raw: Any) -> int | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, Decimal)):
        n = int(raw)
        if 1500 <= n <= 2150 and Decimal(raw) == n:
            return n
        return None
    s = str(raw).strip()
    m_iso = re.match(r"^(\d{4})-\d{2}(?:-\d{2})?$", s)
    if m_iso:
        return int(m_iso.group(1))
    m_yr = re.match(r"^(1[5-9]\d{2}|20\d{2}|21\d{2})(?:\.0+)?$", s)
    if m_yr:
        return int(m_yr.group(1))
    return None


def _sentence_around_ref(doc_text: str, start: int, end: int) -> str:
    if not doc_text:
        return ""
    left = max(
        doc_text.rfind(".", 0, start),
        doc_text.rfind(";", 0, start),
        doc_text.rfind("\n", 0, start),
    )
    right_candidates = [pos for pos in (doc_text.find(".", end), doc_text.find(";", end), doc_text.find("\n", end)) if pos != -1]
    right = min(right_candidates) if right_candidates else len(doc_text)
    return doc_text[left + 1 : right].strip()


def _extract_event_tokens_from_sentence(sentence: str, subject: str | None) -> tuple[str, ...]:
    if not sentence:
        return ()
    subj_words = set()
    if subject:
        subj_words = {_norm_token(t) for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", subject)}
    age_predicate_words = {"tenia", "tiene", "tener", "edad", "age", "old"}
    out: list[str] = []
    for raw_tok in re.findall(r"[A-Za-zÀ-ÿ]+", sentence):
        nt = _norm_token(raw_tok)
        if (
            len(nt) < 4
            or nt in _FUNCTION_WORDS
            or nt in _EPISTEMIC_META_TOKENS
            or nt in _COMMON_STOPWORDS
            or nt in _UNIT_WORD_MAP
            or nt in subj_words
            or nt in age_predicate_words
        ):
            continue
        out.append(nt)
    return tuple(dict.fromkeys(out))


def build_supported_graph_relations(
    graph: JustificationGraph,
    documents: Mapping[str, SourceDocument] | None = None,
    *,
    include_conflict_alternatives: bool = False,
) -> list[GraphRelationTuple]:
    """Build normalized structured relation tuples ``(subject, slot_family, value, unit, temporal_anchor)`` from the graph."""
    conc_id = graph.conclusion_id
    anc_ids = graph.ancestor_claims(conc_id)
    node_ids = [conc_id, *sorted(aid for aid in anc_ids if aid != conc_id)]
    seen_ids = set(node_ids)
    docs_map = documents or {}
    for nid in sorted(graph.claims):
        if nid in seen_ids:
            continue
        cand = graph.claims[nid]
        if cand.structural_reliability is not None and cand.structural_reliability < 0.5:
            continue
        if cand.evidence and (
            cand.kind == "extraction"
            or (
                include_conflict_alternatives
                and all(
                    docs_map.get(ev.document_id) is None
                    or getattr(docs_map.get(ev.document_id), "attribution", None) == "source_statement"
                    for ev in cand.evidence
                )
            )
        ):
            node_ids.append(nid)
            seen_ids.add(nid)

    # First pass: inspect temporal inference steps for anchor/reference years
    node_anchors: dict[str, int] = {}
    extra_tuples: list[GraphRelationTuple] = []
    conc = graph.claims.get(conc_id)
    default_subj = _norm_phrase(conc.subject) if conc is not None else None
    if not default_subj:
        for nid in node_ids:
            if nid in graph.claims and graph.claims[nid].subject:
                default_subj = _norm_phrase(graph.claims[nid].subject)
                break

    for step in graph.steps:
        params = step.parameters or {}
        anchor_yr = _extract_year_int(params.get("anchor_date") or params.get("anchor_year"))
        ref_yr = _extract_year_int(params.get("reference_date") or params.get("reference_year"))
        if anchor_yr is None:
            for pid in step.premises:
                pn = graph.claims.get(pid)
                if pn is not None and (pn.predicate or "").lower() in _DATE_PREDICATE_NAMES:
                    y = _extract_year_int(pn.value)
                    if y is not None:
                        anchor_yr = y
                        break
        if anchor_yr is not None:
            for pid in step.premises:
                node_anchors.setdefault(pid, anchor_yr)
        step_subj = _norm_phrase(graph.claims[step.conclusion_id].subject) if step.conclusion_id in graph.claims else default_subj
        if ref_yr is not None:
            cnode = graph.claims.get(step.conclusion_id)
            if cnode is not None and isinstance(cnode.value, (int, float, Decimal)) and not cnode.approximate:
                node_anchors.setdefault(step.conclusion_id, ref_yr)
            elif cnode is not None and (cnode.predicate or "").lower() in {"age", "anchor_age"} and not isinstance(cnode.value, Mapping):
                # Point-valued derived age for reference_year applies to ref_yr even when marked approximate
                node_anchors.setdefault(step.conclusion_id, ref_yr)
            extra_tuples.append(GraphRelationTuple(
                claim_id=f"{step.step_id}::reference_date",
                subject=step_subj,
                predicate="reference_date",
                slot_family="calendar_date",
                value=Decimal(ref_yr),
                interval=None,
                unit=None,
                temporal_anchor=ref_yr,
            ))
            if anchor_yr is not None and ref_yr >= anchor_yr:
                extra_tuples.append(GraphRelationTuple(
                    claim_id=f"{step.step_id}::elapsed_years",
                    subject=step_subj,
                    predicate="elapsed_duration",
                    slot_family="duration_or_age",
                    value=Decimal(ref_yr - anchor_yr),
                    interval=None,
                    unit="years",
                    temporal_anchor=ref_yr,
                ))
        for k_dt in ("start_date", "end_date"):
            yr_k = _extract_year_int(params.get(k_dt))
            if yr_k is not None:
                extra_tuples.append(GraphRelationTuple(
                    claim_id=f"{step.step_id}::{k_dt}",
                    subject=step_subj,
                    predicate=k_dt,
                    slot_family="calendar_date",
                    value=Decimal(yr_k),
                    interval=None,
                    unit=None,
                    temporal_anchor=yr_k,
                ))

    tuples: list[GraphRelationTuple] = []
    for nid in node_ids:
        node = graph.claims.get(nid)
        if node is None:
            continue
        pred_low = (node.predicate or "").lower().strip()
        raw_unit = node.unit
        norm_unit = _UNIT_WORD_MAP.get(_norm_token(raw_unit), (raw_unit, ""))[0] if raw_unit else None
        val_dec = _to_dec(node.value)
        interval_dec: tuple[Decimal, Decimal] | None = None
        if val_dec is None and isinstance(node.value, Mapping) and "min" in node.value and "max" in node.value:
            lo_d, hi_d = _to_dec(node.value["min"]), _to_dec(node.value["max"])
            if lo_d is not None and hi_d is not None:
                interval_dec = (lo_d, hi_d)

        if pred_low in _DATE_PREDICATE_NAMES:
            slot_family = "calendar_date"
            norm_unit = None
            yr_val = _extract_year_int(node.value)
            if val_dec is None and yr_val is not None:
                val_dec = Decimal(yr_val)
            anchor = yr_val if yr_val is not None else node_anchors.get(nid)
        elif pred_low in _AGE_DURATION_PREDICATE_NAMES or (norm_unit in _TEMPORAL_UNITS):
            slot_family = "duration_or_age"
            anchor = node_anchors.get(nid)
        elif norm_unit in _MONETARY_UNITS or pred_low in _MONETARY_PREDICATE_NAMES:
            slot_family = "monetary_amount"
            anchor = node_anchors.get(nid)
        elif norm_unit in _RATE_UNITS:
            slot_family = "percentage_or_rate"
            anchor = node_anchors.get(nid)
        else:
            yr_val = _extract_year_int(node.value) if norm_unit is None else None
            if yr_val is not None:
                slot_family = "calendar_date"
                anchor = yr_val
            else:
                slot_family = "quantity"
                anchor = node_anchors.get(nid)

        subj_norm = _norm_phrase(node.subject) or default_subj
        source_ent: str | None = None
        ev_tokens: list[str] = []
        for ev in node.evidence:
            doc_obj = (documents or {}).get(ev.document_id)
            if doc_obj is None:
                continue
            m_src = _SOURCE_LEAD_ENTITY_RE.match(doc_obj.text)
            if m_src:
                cand_src = _norm_phrase(m_src.group(1))
                if cand_src and cand_src != subj_norm:
                    source_ent = cand_src
            if node.kind == "extraction" and anchor is not None:
                sent = _sentence_around_ref(doc_obj.text, ev.start, ev.end)
                ev_tokens.extend(_extract_event_tokens_from_sentence(sent, node.subject))

        tuples.append(GraphRelationTuple(
            claim_id=nid,
            subject=subj_norm,
            predicate=pred_low or None,
            slot_family=slot_family,
            value=val_dec,
            interval=interval_dec,
            unit=norm_unit,
            temporal_anchor=anchor,
            source_entity=source_ent,
            event_tokens=tuple(dict.fromkeys(ev_tokens)),
        ))

    tuples.extend(extra_tuples)
    return tuples


def _units_compatible(obs_unit: str | None, graph_unit: str | None, unit_registry: UnitRegistry | None) -> bool:
    if obs_unit is None or graph_unit is None:
        return True
    if _norm_token(obs_unit) == _norm_token(graph_unit):
        return True
    o_canon = _UNIT_WORD_MAP.get(_norm_token(obs_unit), (obs_unit, ""))[0]
    g_canon = _UNIT_WORD_MAP.get(_norm_token(graph_unit), (graph_unit, ""))[0]
    if _norm_token(o_canon) == _norm_token(g_canon):
        return True
    # Allow annualized currency presentation vs currency (e.g. USD_per_year vs USD)
    if {o_canon, g_canon} == {"USD", "USD_per_year"}:
        return True
    # Allow annualized percent presentation vs percent (e.g. "5.80%" vs percent_per_annum),
    # while keeping percentage_point vs percent strictly incompatible.
    if {o_canon, g_canon} in (
        {"percent", "percent_per_annum"},
        {"basis_point", "basis_points_per_annum"},
    ):
        return True
    # Explicitly forbid equating calendar days and business days
    if {o_canon, g_canon} == {"calendar_days", "business_days"}:
        return False
    if unit_registry is not None:
        try:
            return unit_registry.lookup(o_canon).dimension == unit_registry.lookup(g_canon).dimension
        except Exception:
            pass
    # Compatible within general temporal units excluding calendar vs business day confusion
    if o_canon in _TEMPORAL_UNITS and g_canon in _TEMPORAL_UNITS:
        return True
    return False


def _find_entity_spans(text: str, known_entities: Sequence[str]) -> list[tuple[int, int, str]]:
    norm_text = _strip_accents(text.lower())
    raw_spans: list[tuple[int, int, str]] = []
    # Match longer entity names first to avoid prefix shadows
    for ent in sorted(set(known_entities), key=len, reverse=True):
        if len(ent) < 3:
            continue
        for m in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(ent)}(?![A-Za-z0-9_])", norm_text):
            if any(not (m.end() <= s or m.start() >= e) for s, e, _ in raw_spans):
                continue
            raw_spans.append((m.start(), m.end(), ent))
    raw_spans.sort(key=lambda item: item[0])
    return raw_spans


def _partition_segment_by_entities(segment: str, leaf_entities: Sequence[str]) -> list[str]:
    spans = _find_entity_spans(segment, leaf_entities)
    distinct_ents = []
    for _, _, ent in spans:
        if ent not in distinct_ents:
            distinct_ents.append(ent)
    if len(distinct_ents) < 2:
        return [segment]

    # Collect unique anchor occurrences when the entity switches
    boundaries: list[tuple[int, int, str]] = []
    prev_ent = None
    for s, e, ent in spans:
        if ent != prev_ent:
            boundaries.append((s, e, ent))
            prev_ent = ent
    if len(boundaries) < 2:
        return [segment]

    # Check if there is at least one numeric assertion between consecutive distinct entities
    pieces: list[str] = []
    cur_start = 0
    for idx in range(len(boundaries) - 1):
        _, e_cur, _ = boundaries[idx]
        s_next, _, _ = boundaries[idx + 1]
        between = segment[e_cur:s_next]
        # Only split if left chunk already contains a number or a distinct clause boundary
        left_chunk = segment[cur_start:s_next]
        if _NUMBER_TOKEN_RE.search(between) or _NUMBER_TOKEN_RE.search(segment[cur_start : boundaries[idx][0]]):
            # Prefer cutting right after any coordinator/punctuation just before s_next
            cut = s_next
            m_sep = re.search(r"(?:[;,]|\s+(?:y|e|and|pero|but|mientras\s+que|whereas|while))\s*$", segment[cur_start:s_next], re.IGNORECASE)
            if m_sep:
                cut = cur_start + m_sep.start()
            part = segment[cur_start:cut].strip(" ,;:")
            if part:
                pieces.append(part)
            cur_start = s_next
    tail = segment[cur_start:].strip(" ,;:")
    if tail:
        pieces.append(tail)
    return pieces if len(pieces) >= 2 else [segment]


def _classify_number_slot(
    seg: str,
    start: int,
    end: int,
    val: Decimal,
    *,
    is_range_lo_or_hi: bool = False,
) -> tuple[str, str | None]:
    before = seg[:start]
    after = seg[end:]

    # 1. Currency prefix ("$500")
    m_cur_pre = re.search(r"([$€£])\s*$", before)
    if m_cur_pre:
        sym = m_cur_pre.group(1)
        canon, slot = _UNIT_WORD_MAP[sym]
        return slot, canon

    # 2. Explicit measurement unit immediately after ("13 años", "500 USD", "10 %", "25 bps", "0.25 pp")
    m_unit = re.match(
        r"^\s*(?:de\s+)?(d[ií]as\s+h[aá]biles|d[ií]as\s+laborables|d[ií]as\s+naturales|"
        r"business\s+days?|working\s+days?|calendar\s+days?|por\s+ciento|"
        r"percentage\s+points?\s+per\s+annum|basis\s+points?\s+per\s+annum|"
        r"percentage\s+points?|basis\s+points?|puntos?\s+porcentuales?|puntos?\s+b[aá]sicos?|"
        r"usd\s+per\s+year|dollars?\s+per\s+year|per\s+annum|"
        r"[A-Za-zÀ-ÿ]+|[$€£%])",
        after,
        re.IGNORECASE,
    )
    if m_unit:
        raw_u = re.sub(r"\s+", " ", m_unit.group(1)).strip()
        nu = _norm_token(raw_u)
        if nu in _UNIT_WORD_MAP:
            canon, slot = _UNIT_WORD_MAP[nu]
            return slot, canon

    # 3. Date / temporal anchor preposition before ("en 2026", "en 13", "era 1933")
    if _DATE_PRE_CONTEXT_RE.search(before):
        return "calendar_date", None

    # 4. Shared trailing unit via coordination ("400 y 600 USD")
    m_coord_u = re.match(
        r"^\s*(?:y|e|a|and|to|-)\s+-?\d+(?:[.,]\d+)?\s*(d[ií]as\s+h[aá]biles|d[ií]as\s+naturales|[A-Za-zÀ-ÿ]+|[$€£%])",
        after,
        re.IGNORECASE,
    )
    if m_coord_u:
        nu2 = _norm_token(re.sub(r"\s+", " ", m_coord_u.group(1)))
        if nu2 in _UNIT_WORD_MAP:
            canon, slot = _UNIT_WORD_MAP[nu2]
            return slot, canon

    # 5. Non-unit counted noun immediately after ("106 acciones")
    m_noun = re.match(r"^\s+(?:de\s+)?([A-Za-zÀ-ÿ]{3,})\b", after, re.IGNORECASE)
    if m_noun:
        noun_raw = m_noun.group(1)
        noun_norm = _norm_token(noun_raw)
        if (
            noun_norm not in _UNIT_WORD_MAP
            and noun_norm not in _FUNCTION_WORDS
            and noun_norm not in _EPISTEMIC_META_TOKENS
            and noun_norm not in _NON_COUNTED_POST_NUM_WORDS
            and not _COPULA_OR_PRED_VERB_RE.fullmatch(noun_raw)
        ):
            # Check not preceded by "de las/los/la/el" (e.g. "1000 de las facturas")
            if not re.match(r"^\s+de\s+(?:el|la|los|las|this|the)\b", after, re.IGNORECASE):
                return f"counted_noun:{_stem_key(noun_norm)}", noun_norm

    # 6. Unambiguous predicate linking copula (only if seg has a linking verb & predicate keyword)
    seg_norm = _strip_accents(seg.lower())
    if _COPULA_OR_PRED_VERB_RE.search(seg) and len(_extract_clause_numbers(seg)) == 1:
        if any(k in seg_norm for k in ("edad", "age")):
            return "duration_or_age", "years"
        if any(k in seg_norm for k in ("reembolso", "refund", "importe", "monto", "total", "subtotal", "limite", "cap", "ingreso", "revenue", "cargo", "charge")):
            return "monetary_amount", None

    if is_range_lo_or_hi:
        yr_chk = _extract_year_int(val)
        if yr_chk is not None:
            return "calendar_date", None

    return "untyped_number", None


def _verify_subclause_relations(
    sub_cl: str,
    graph_relations: list[GraphRelationTuple],
    default_subj: str | None,
    verified_raw_equations: Sequence[str],
    *,
    unit_registry: UnitRegistry | None = None,
    semantic_evaluator: Any = None,
) -> tuple[list[ResponseRelationTuple], list[dict[str, Any]]]:
    """Verify relational structure ``(subject, slot_family, value, unit, temporal_anchor)`` inside a clause."""
    cleaned = sub_cl
    for raw_eq in sorted(verified_raw_equations, key=len, reverse=True):
        if raw_eq and raw_eq in cleaned:
            cleaned = cleaned.replace(raw_eq, " ")

    # Specific leaf subjects and source entities (exclude purely aggregate conclusion subject when finer leaf subjects exist)
    all_ent_names: list[str] = []
    leaf_ent_names: list[str] = []
    for gr in graph_relations:
        for nm in (gr.subject, gr.source_entity):
            if nm and nm not in all_ent_names:
                all_ent_names.append(nm)
                if gr.claim_id != graph_relations[0].claim_id or len({g.subject for g in graph_relations if g.subject}) == 1:
                    leaf_ent_names.append(nm)
    # Build set of entity strings present on tuples (keep conclusion aggregate/container subject out when finer leaf/source entities exist)
    def_subj_norm = _norm_phrase(default_subj)
    distinct_subjects = {g.subject for g in graph_relations if g.subject}
    has_source_entities = any(bool(g.source_entity) for g in graph_relations)
    specific_subjects: set[str] = set()
    for gr in graph_relations:
        if gr.source_entity:
            specific_subjects.add(gr.source_entity)
        if gr.subject and gr.subject != def_subj_norm:
            specific_subjects.add(gr.subject)
        elif gr.subject and not has_source_entities and len(distinct_subjects) > 1 and gr.claim_id != graph_relations[0].claim_id:
            specific_subjects.add(gr.subject)
    # If all nodes share a single subject (e.g. Warren Buffet), include it in entity detection
    if not specific_subjects and def_subj_norm:
        specific_subjects.add(def_subj_norm)
    specific_subjects.discard("")

    partition_targets = sorted(specific_subjects | {e for e in all_ent_names if e}, key=len, reverse=True)
    raw_segments = _partition_segment_by_entities(cleaned, partition_targets)
    segments: list[str] = []
    _agg_tail_split_re = re.compile(
        r",\s*(?=(?:por\s+lo\s+que|as[ií]\s+que|de\s+modo\s+que|de\s+manera\s+que|"
        r"sumando|totalizando|lo\s+que\s+suma|therefore|totalling|totaling|"
        r"(?:el|la|los|las|un|una)?\s*(?:subtotal|total)\b))",
        re.IGNORECASE,
    )
    for rseg in raw_segments:
        sub_parts = _agg_tail_split_re.split(rseg)
        if len(sub_parts) > 1 and all(re.search(r"\d", p) for p in sub_parts):
            segments.extend(p.strip() for p in sub_parts if p.strip())
        else:
            segments.append(rseg)

    # Precompute exclusive event tokens mapped to a single temporal_anchor across graph tuples
    token_anchors: dict[str, set[int]] = {}
    for gr in graph_relations:
        if gr.temporal_anchor is not None:
            for tok in gr.event_tokens:
                token_anchors.setdefault(tok, set()).add(gr.temporal_anchor)
    exclusive_event_tokens: dict[str, int] = {
        tok: next(iter(ancs)) for tok, ancs in token_anchors.items() if len(ancs) == 1
    }

    extracted_rels: list[ResponseRelationTuple] = []
    issues: list[dict[str, Any]] = []
    def_subj_norm = _norm_phrase(default_subj)

    for seg in segments:
        seg_spans = _find_entity_spans(seg, partition_targets)
        # If the first recognized entity span sits after a comma-separated numeric clause (e.g., an unrecognized sibling entity),
        # do not backward-bind the earlier clause's number to this later entity.
        if len(seg_spans) == 1 and seg_spans[0][0] > 0:
            before_ent = seg[: seg_spans[0][0]]
            if "," in before_ent and _NUMBER_TOKEN_RE.search(before_ent):
                seg = seg[seg_spans[0][0] :].strip()
                seg_spans = _find_entity_spans(seg, partition_targets)
        seg_matched_ents = list(dict.fromkeys(ent for _, _, ent in seg_spans))
        specific_matched = [e for e in seg_matched_ents if e in specific_subjects]
        if len(specific_matched) == 1:
            seg_subject: str | None = specific_matched[0]
            multi_entity = False
        elif len(specific_matched) > 1:
            seg_subject = def_subj_norm
            multi_entity = True
        else:
            seg_subject = seg_matched_ents[0] if len(seg_matched_ents) == 1 else def_subj_norm
            multi_entity = False

        has_agg_kw = bool(_AGGREGATE_PHRASE_RE.search(_strip_accents(seg.lower())))
        conc_subj_norm = def_subj_norm
        is_agg_seg = bool(
            has_agg_kw
            and (multi_entity or not specific_matched or seg_subject == conc_subj_norm)
        )

        # Extract range spans first so individual bounds aren't double-checked as bare numbers
        range_entries: list[tuple[int, int, Decimal, Decimal, str, str | None]] = []
        for m_r in _RANGE_PATTERN_RE.finditer(seg):
            g = m_r.groups()
            lo_r, hi_r = None, None
            if g[0] is not None and g[1] is not None:
                lo_r, hi_r = _to_dec(g[0]), _to_dec(g[1])
            elif g[2] is not None and g[3] is not None:
                lo_r, hi_r = _to_dec(g[2]), _to_dec(g[3])
            elif g[4] is not None and g[5] is not None:
                lo_r, hi_r = _to_dec(g[4]), _to_dec(g[5])
            elif g[6] is not None and g[7] is not None:
                ctr, tol = _to_dec(g[6]), _to_dec(g[7])
                if ctr is not None and tol is not None and tol >= 0:
                    lo_r, hi_r = ctr - tol, ctr + tol
            if lo_r is not None and hi_r is not None and lo_r <= hi_r:
                r_slot, r_unit = _classify_number_slot(seg, m_r.start(), m_r.end(), hi_r, is_range_lo_or_hi=True)
                range_entries.append((m_r.start(), m_r.end(), lo_r, hi_r, r_slot, r_unit))

        num_entries: list[tuple[int, int, Decimal, str, str | None]] = []
        for m_n in _NUMBER_TOKEN_RE.finditer(seg):
            if any(rs <= m_n.start() and m_n.end() <= re_ for rs, re_, *_ in range_entries):
                continue
            prefix_ctx = seg[max(0, m_n.start() - 28):m_n.start()]
            if re.search(
                r"(?:\b(?:section|sec\.?|article|art\.?|clause|step|steps|rank|secci[oó]n|art[ií]culo|cl[aá]usula|paso|pasos|rango)\s*$|\b(?:steps?|pasos?)\s+\d+\s*(?:,|and|y)\s*$)",
                prefix_ctx,
                re.IGNORECASE,
            ):
                continue
            d_val = _to_dec(m_n.group(1))
            if d_val is None:
                continue
            slot, unit = _classify_number_slot(seg, m_n.start(), m_n.end(), d_val)
            num_entries.append((m_n.start(), m_n.end(), d_val, slot, unit))

        # Determine calendar anchor year(s) mentioned inside this segment
        seg_anchor_years: list[int] = []
        for _, _, d_val, slot, _ in num_entries:
            if slot == "calendar_date" and d_val == int(d_val):
                seg_anchor_years.append(int(d_val))
        unique_seg_years = set(seg_anchor_years)
        single_anchor_year = next(iter(unique_seg_years)) if len(unique_seg_years) == 1 else None

        # Check event-token temporal mismatch inside this segment (e.g. "compró su primera acción en 2026")
        if single_anchor_year is not None and exclusive_event_tokens:
            seg_ev_toks = [
                _norm_token(t)
                for t in re.findall(r"[A-Za-zÀ-ÿ]+", seg)
                if _norm_token(t) in exclusive_event_tokens
            ]
            # Require >=2 distinct event tokens whose anchor disagrees when no duration/age conclusion explains single_anchor_year
            mismatched_ev_toks = [t for t in dict.fromkeys(seg_ev_toks) if exclusive_event_tokens[t] != single_anchor_year]
            has_age_for_year = any(s == "duration_or_age" for _, _, _, s, _ in num_entries)
            if len(mismatched_ev_toks) >= 2 and not has_age_for_year:
                expected_yr = exclusive_event_tokens[mismatched_ev_toks[0]]
                reason = (
                    f"Event tokens {mismatched_ev_toks} belong to temporal anchor {expected_yr}, "
                    f"not {single_anchor_year}, in '{seg}'."
                )
                rel = ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family="event_temporal_anchor",
                    value=Decimal(single_anchor_year),
                    unit=None,
                    temporal_anchor=single_anchor_year,
                    status="refuted",
                    reason=reason,
                )
                extracted_rels.append(rel)
                issues.append({
                    "code": "RELATIONAL_VALUE_MISMATCH",
                    "status": "refuted",
                    "reason": reason,
                    "severity": "error",
                    "clause": sub_cl,
                    "relation": rel.to_dict(),
                })

        # Select entity-filtered relation pool when seg_subject targets a specific leaf/source entity
        entity_specific_mode = (
            seg_subject is not None
            and seg_subject in specific_subjects
            and len(distinct_subjects) > 1
            and not is_agg_seg
            and not multi_entity
        ) or (
            seg_subject is not None
            and any(gr.source_entity == seg_subject for gr in graph_relations)
            and not multi_entity
        )
        scoped_pool = (
            [gr for gr in graph_relations if gr.subject == seg_subject or gr.source_entity == seg_subject]
            if entity_specific_mode
            else graph_relations
        )

        # Verify range entries
        for _, _, lo_r, hi_r, r_slot, r_unit in range_entries:
            matched_tuple: GraphRelationTuple | None = None
            for gr in scoped_pool:
                if gr.interval is not None:
                    if abs(gr.interval[0] - lo_r) <= Decimal("0.01") and abs(gr.interval[1] - hi_r) <= Decimal("0.01"):
                        if _units_compatible(r_unit, gr.unit, unit_registry) and (
                            r_slot == "untyped_number" or r_slot == gr.slot_family
                        ):
                            matched_tuple = gr
                            break
            if matched_tuple is not None:
                extracted_rels.append(ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family=r_slot,
                    value=None,
                    unit=r_unit,
                    temporal_anchor=single_anchor_year,
                    status="verified",
                    matched_claim_id=matched_tuple.claim_id,
                ))
            else:
                reason = f"Interval [{lo_r}, {hi_r}] ({r_slot}/{r_unit}) in '{seg}' does not match supported graph relations."
                rel = ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family=r_slot,
                    value=None,
                    unit=r_unit,
                    temporal_anchor=single_anchor_year,
                    status="refuted",
                    reason=reason,
                )
                extracted_rels.append(rel)
                issues.append({
                    "code": "RELATIONAL_VALUE_MISMATCH",
                    "status": "refuted",
                    "reason": reason,
                    "severity": "error",
                    "clause": sub_cl,
                    "relation": rel.to_dict(),
                })

        # Verify each point numeric mention inside seg
        for _, _, val, obs_slot, obs_unit in num_entries:
            # 1) Untyped bare numbers cannot silently count as verified
            if obs_slot == "untyped_number":
                sem_status = "unknown"
                if (
                    semantic_evaluator is not None
                    and getattr(semantic_evaluator, "enabled", False)
                    and getattr(semantic_evaluator, "callable_evaluator", None) is not None
                ):
                    try:
                        sem_res = semantic_evaluator.callable_evaluator(
                            check="relational_correspondence",
                            clause=seg,
                            subject=seg_subject,
                            value=str(val),
                            supported_relations=[gr.to_dict() for gr in scoped_pool],
                        )
                        if isinstance(sem_res, Mapping) and sem_res.get("status") in ("verified", "refuted", "unknown"):
                            sem_status = str(sem_res["status"])
                    except Exception:
                        sem_status = "unknown"
                if sem_status == "verified":
                    extracted_rels.append(ResponseRelationTuple(
                        clause_text=seg,
                        subject=seg_subject,
                        slot_family=obs_slot,
                        value=val,
                        unit=obs_unit,
                        temporal_anchor=single_anchor_year,
                        status="verified",
                        reason="Resolved by semantic evaluator.",
                    ))
                elif sem_status == "refuted":
                    reason = f"Untyped relation for value {val} in '{seg}' refuted semantically."
                    rel = ResponseRelationTuple(
                        clause_text=seg,
                        subject=seg_subject,
                        slot_family=obs_slot,
                        value=val,
                        unit=obs_unit,
                        temporal_anchor=single_anchor_year,
                        status="refuted",
                        reason=reason,
                    )
                    extracted_rels.append(rel)
                    issues.append({
                        "code": "RELATIONAL_VALUE_MISMATCH",
                        "status": "refuted",
                        "reason": reason,
                        "severity": "error",
                        "clause": sub_cl,
                        "relation": rel.to_dict(),
                    })
                else:
                    reason = (
                        f"Numeric relation for value {val} in '{seg}' lacks explicit role/unit binding "
                        "and requires semantic verification."
                    )
                    rel = ResponseRelationTuple(
                        clause_text=seg,
                        subject=seg_subject,
                        slot_family=obs_slot,
                        value=val,
                        unit=obs_unit,
                        temporal_anchor=single_anchor_year,
                        status="unknown",
                        reason=reason,
                    )
                    extracted_rels.append(rel)
                    issues.append({
                        "code": "RELATIONAL_CORRESPONDENCE_UNRESOLVED",
                        "status": "unknown",
                        "reason": reason,
                        "severity": "warning",
                        "clause": sub_cl,
                        "relation": rel.to_dict(),
                    })
                continue

            # 2) Typed or counted_noun numeric assertion: match against scoped_pool
            def _val_matches_gr(v_obs: Decimal, u_obs: str | None, gr: GraphRelationTuple) -> bool:
                if gr.value is None:
                    return False
                if abs(gr.value - v_obs) <= Decimal("0.01"):
                    return True
                if unit_registry is not None and u_obs and gr.unit:
                    try:
                        uo = unit_registry.lookup(_UNIT_WORD_MAP.get(_norm_token(u_obs), (u_obs, ""))[0])
                        ug = unit_registry.lookup(_UNIT_WORD_MAP.get(_norm_token(gr.unit), (gr.unit, ""))[0])
                        if uo.dimension == ug.dimension:
                            bo = v_obs * Decimal(uo.factor)
                            bg = gr.value * Decimal(ug.factor)
                            return abs(bo - bg) <= Decimal("0.0001")
                    except Exception:
                        pass
                return False

            val_candidates = [gr for gr in scoped_pool if _val_matches_gr(val, obs_unit, gr)]

            if not val_candidates:
                if entity_specific_mode:
                    reason = (
                        f"Entity '{seg_subject}' is paired with value {val} ({obs_slot}) in '{seg}', "
                        f"which conflicts with supported relations for '{seg_subject}'."
                    )
                    code = "RELATIONAL_VALUE_MISMATCH"
                else:
                    reason = (
                        f"Value {val} ({obs_slot}) in '{seg}' does not match any supported graph relation "
                        f"for '{seg_subject or 'conclusion'}'."
                    )
                    code = "RELATIONAL_VALUE_MISMATCH"
                rel = ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family=obs_slot,
                    value=val,
                    unit=obs_unit,
                    temporal_anchor=single_anchor_year,
                    status="refuted",
                    reason=reason,
                )
                extracted_rels.append(rel)
                issues.append({
                    "code": code,
                    "status": "refuted",
                    "reason": reason,
                    "severity": "error",
                    "clause": sub_cl,
                    "relation": rel.to_dict(),
                })
                continue

            # Check counted_noun reassignment ("106 acciones") vs node slot/unit
            if obs_slot.startswith("counted_noun:"):
                noun_stem = obs_slot.split(":", 1)[1]
                noun_ok = [
                    gr for gr in val_candidates
                    if gr.slot_family == "quantity"
                    and (
                        (gr.unit and _stem_key(_norm_token(gr.unit)) == noun_stem)
                        or (gr.predicate and noun_stem in _norm_token(gr.predicate))
                    )
                ]
                if not noun_ok:
                    reason = (
                        f"Value {val} is bound to counted noun '{obs_unit}' in '{seg}', "
                        f"which mismatches graph role(s) {[gr.slot_family for gr in val_candidates]}."
                    )
                    rel = ResponseRelationTuple(
                        clause_text=seg,
                        subject=seg_subject,
                        slot_family=obs_slot,
                        value=val,
                        unit=obs_unit,
                        temporal_anchor=single_anchor_year,
                        status="refuted",
                        reason=reason,
                    )
                    extracted_rels.append(rel)
                    issues.append({
                        "code": "RELATIONAL_VALUE_MISMATCH",
                        "status": "refuted",
                        "reason": reason,
                        "severity": "error",
                        "clause": sub_cl,
                        "relation": rel.to_dict(),
                    })
                    continue
                val_candidates = noun_ok

            # Check slot_family role compatibility ("1933 años" vs calendar_date 1933, "en 13" vs duration_or_age 13)
            slot_ok = [gr for gr in val_candidates if gr.slot_family == obs_slot]
            if not slot_ok:
                reason = (
                    f"Value {val} is used in role '{obs_slot}' in '{seg}', "
                    f"whereas the graph supports role(s) {[gr.slot_family for gr in val_candidates]}."
                )
                rel = ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family=obs_slot,
                    value=val,
                    unit=obs_unit,
                    temporal_anchor=single_anchor_year,
                    status="refuted",
                    reason=reason,
                )
                extracted_rels.append(rel)
                issues.append({
                    "code": "RELATIONAL_VALUE_MISMATCH",
                    "status": "refuted",
                    "reason": reason,
                    "severity": "error",
                    "clause": sub_cl,
                    "relation": rel.to_dict(),
                })
                continue

            # Check unit compatibility
            unit_ok = [gr for gr in slot_ok if _units_compatible(obs_unit, gr.unit, unit_registry)]
            if not unit_ok:
                reason = (
                    f"Unit '{obs_unit}' on value {val} in '{seg}' is incompatible with "
                    f"graph unit(s) {[gr.unit for gr in slot_ok]}."
                )
                rel = ResponseRelationTuple(
                    clause_text=seg,
                    subject=seg_subject,
                    slot_family=obs_slot,
                    value=val,
                    unit=obs_unit,
                    temporal_anchor=single_anchor_year,
                    status="refuted",
                    reason=reason,
                )
                extracted_rels.append(rel)
                issues.append({
                    "code": "RELATIONAL_VALUE_MISMATCH",
                    "status": "refuted",
                    "reason": reason,
                    "severity": "error",
                    "clause": sub_cl,
                    "relation": rel.to_dict(),
                })
                continue

            # Check temporal_anchor compatibility when a non-date assertion co-occurs with a single year
            if obs_slot != "calendar_date" and single_anchor_year is not None:
                anchor_ok = [
                    gr for gr in unit_ok
                    if gr.temporal_anchor is None or gr.temporal_anchor == single_anchor_year
                ]
                if not anchor_ok:
                    expected_anchors = sorted({gr.temporal_anchor for gr in unit_ok if gr.temporal_anchor is not None})
                    reason = (
                        f"Assertion for value {val} ({obs_slot}) is anchored to year {single_anchor_year} in '{seg}', "
                        f"but graph supports temporal anchor(s) {expected_anchors}."
                    )
                    rel = ResponseRelationTuple(
                        clause_text=seg,
                        subject=seg_subject,
                        slot_family=obs_slot,
                        value=val,
                        unit=obs_unit,
                        temporal_anchor=single_anchor_year,
                        status="refuted",
                        reason=reason,
                    )
                    extracted_rels.append(rel)
                    issues.append({
                        "code": "RELATIONAL_VALUE_MISMATCH",
                        "status": "refuted",
                        "reason": reason,
                        "severity": "error",
                        "clause": sub_cl,
                        "relation": rel.to_dict(),
                    })
                    continue
                unit_ok = anchor_ok

            matched_gr = unit_ok[0]
            extracted_rels.append(ResponseRelationTuple(
                clause_text=seg,
                subject=seg_subject,
                slot_family=obs_slot,
                value=val,
                unit=obs_unit,
                temporal_anchor=single_anchor_year,
                status="verified",
                matched_claim_id=matched_gr.claim_id,
            ))

    return extracted_rels, issues


def _is_graph_conclusion_negative(graph: JustificationGraph) -> bool:
    conc = graph.claims[graph.conclusion_id]
    if isinstance(conc.value, bool) and conc.value is False:
        return True
    if isinstance(conc.scope, Mapping) and str(conc.scope.get("polarity", "")).lower() == "negative":
        return True
    if conc.predicate and conc.predicate.lower().startswith("not_"):
        return True
    return False


_OPINION_MARKER_RE = re.compile(
    r"\b(?:opinion|interpretive|interpretation|in\s+our\s+view|we\s+believe|"
    r"en\s+nuestra\s+opini[oó]n|interpretaci[oó]n|nota\s+interpretativa)\b",
    re.IGNORECASE,
)


def _clause_binds_to_graph_claim(
    sub_cl: str,
    cl_nums: tuple[Decimal, ...],
    graph: JustificationGraph,
    *,
    role: str,
    has_material_conflict: bool,
    doc_q_exact: set[str] | frozenset[str] = (),
) -> tuple[bool, str | None]:
    """Verify every non-formula/non-disclaimer clause binds to a verified graph claim or marked opinion."""
    if role in ("formula", "disclaimer"):
        return True, None
    if role == "conflict_disclosure" and has_material_conflict:
        return True, None
    if not cl_nums and all(
        _norm_token(tok) in _FUNCTION_WORDS
        or _norm_token(tok) in _EPISTEMIC_META_TOKENS
        or _norm_token(tok) in doc_q_exact
        or len(_norm_token(tok)) < 3
        for tok in re.findall(r"[A-Za-zÀ-ÿ]+", sub_cl)
    ):
        return True, None

    low = sub_cl.lower()
    is_opinion_marked = bool(_OPINION_MARKER_RE.search(sub_cl))

    if is_opinion_marked:
        for cid, node in graph.claims.items():
            if getattr(node, "kind", "") != "opinion":
                continue
            has_premises = any(s.conclusion_id == cid and len(s.premises) > 0 for s in graph.steps)
            if not has_premises:
                continue
            node_subj = str(getattr(node, "subject", "") or "").strip().lower()
            node_text = str(getattr(node, "text", "") or "").strip().lower()
            node_val = str(getattr(node, "value", "") or "").strip().lower()
            if (
                (node_subj and node_subj in low)
                or (node_val and node_val in low)
                or (node_text and any(w in low for w in re.findall(r"[a-z]{4,}", node_text)))
            ):
                return True, None
        return False, "Opinion/interpretive clause has no matching kind='opinion' claim with premises in the graph."

    for cid, node in graph.claims.items():
        if getattr(node, "kind", "") == "opinion":
            continue
        has_grounding = bool(
            getattr(node, "evidence", ())
            or getattr(node, "evidence_unit_ids", ())
            or cid == graph.conclusion_id
        )
        if not has_grounding:
            continue
        node_subj = str(getattr(node, "subject", "") or "").strip().lower()
        if node_subj and node_subj in low:
            return True, None
        node_pred = str(getattr(node, "predicate", "") or "").strip().lower().replace("_", " ")
        if node_pred and len(node_pred) >= 4 and node_pred in low:
            return True, None
        node_conds = getattr(node, "conditions", ()) or ()
        if node_conds:
            for cond in node_conds:
                if isinstance(cond, Mapping):
                    for ck in ("field", "value", "expression", "label"):
                        cv = str(cond.get(ck) or "").strip().lower().replace("_", " ")
                        if cv and len(cv) >= 3 and (cv in low or any(w in low for w in re.findall(r"[a-z]{4,}", cv))):
                            return True, None
                elif isinstance(cond, str):
                    cv = cond.strip().lower().replace("_", " ")
                    if cv and any(w in low for w in re.findall(r"[a-z]{4,}", cv)):
                        return True, None
        raw_val = getattr(node, "value", None)
        if isinstance(raw_val, Mapping) and "value" in raw_val:
            raw_val = raw_val["value"]
        if raw_val is not None:
            val_dec = _to_dec(raw_val)
            if val_dec is not None and any(abs(cn - val_dec) <= Decimal("0.0001") for cn in cl_nums):
                return True, None
            val_str = str(raw_val).strip().lower()
            if val_str and len(val_str) >= 3 and val_str in low:
                return True, None

    return False, (
        f"Clause '{sub_cl}' does not link to any verified factual leaf or marked opinion claim in the graph."
    )


def _word_phrase_in_text(phrase: str, norm_text: str) -> bool:
    p_norm = re.sub(r"\s+", " ", phrase.strip().lower())
    return bool(p_norm) and bool(re.search(r"(?<!\w)" + re.escape(p_norm) + r"(?!\w)", norm_text))


def _extract_register_conflict_pairs(
    corpus_register: Any,
) -> list[tuple[Any, Any, Any, Any]]:
    """Return [(gov_val, gov_unit, sub_val, sub_unit)] for conflict/override corpus obligations."""
    pairs: list[tuple[Any, Any, Any, Any]] = []
    if corpus_register is None:
        return pairs
    for ob in getattr(corpus_register, "obligations", ()) or ():
        params = getattr(ob, "parameters", None) or {}
        kind = str(params.get("corpus_obligation_kind", "") or getattr(ob, "kind", ""))
        if kind == "authority_override":
            pairs.append((
                params.get("winner_value"),
                params.get("winner_unit"),
                params.get("loser_value"),
                params.get("loser_unit"),
            ))
        elif kind == "material_conflict":
            conf_vals = list(params.get("conflicting_values") or ())
            if len(conf_vals) >= 2:
                c0, c1 = conf_vals[0], conf_vals[1]
                r0 = int(c0.get("authority_rank", 0) or 0)
                r1 = int(c1.get("authority_rank", 0) or 0)
                if r0 != r1:
                    win_c, lose_c = (c0, c1) if r0 > r1 else (c1, c0)
                    pairs.append((
                        win_c.get("value"),
                        win_c.get("unit"),
                        lose_c.get("value"),
                        lose_c.get("unit"),
                    ))
                else:
                    pairs.append((c0.get("value"), c0.get("unit"), c1.get("value"), c1.get("unit")))
    return pairs


def _check_register_status_contradiction(
    sub_cl: str,
    *,
    vocabulary: Any = None,
    corpus_register: Any = None,
) -> dict[str, Any] | None:
    """Flag clauses denying discrepancies when the corpus register holds unresolved conflicts/overrides."""
    reg_terms = getattr(vocabulary, "register_status_terms", ()) if vocabulary is not None else ()
    if not reg_terms or corpus_register is None:
        return None
    conflict_obs = [
        ob
        for ob in (getattr(corpus_register, "obligations", ()) or ())
        if (getattr(ob, "parameters", None) or {}).get("corpus_obligation_kind")
        in ("material_conflict", "authority_override")
    ]
    if not conflict_obs:
        return None
    norm_cl = re.sub(r"\s+", " ", sub_cl.strip().lower())
    for term in reg_terms:
        if _word_phrase_in_text(str(term), norm_cl):
            return {
                "code": "REGISTER_STATUS_CONTRADICTED",
                "status": "refuted",
                "reason": (
                    f"Clause '{sub_cl}' asserts '{term}', contradicting the corpus reference "
                    f"register ({len(conflict_obs)} conflict/override obligation(s) present)."
                ),
                "severity": "error",
                "clause": sub_cl,
                "matched_term": str(term),
            }
    return None


def _val_matches_clause(val: Any, sub_cl: str, unit: Any = None) -> bool:
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, Mapping):
        return _value_in_segment_with_unit(val, sub_cl)
    if unit is not None and str(unit).strip():
        return _value_in_segment_with_unit({"value": val, "unit": str(unit)}, sub_cl)
    return _value_in_segment_with_unit(val, sub_cl)


def _collect_concrete_fact_values(
    graph: JustificationGraph,
    corpus_register: Any = None,
) -> list[tuple[Any, str | None]]:
    """Collect concrete scalar/quantity/enum fact values from corpus obligations and non-opinion graph nodes."""
    vals: list[tuple[Any, str | None]] = []
    seen: set[str] = set()

    def _add(val: Any, unit: Any) -> None:
        if val is None or isinstance(val, bool):
            return
        if isinstance(val, Mapping):
            raw = val.get("value")
            if raw is None or isinstance(raw, bool):
                return
            u_str = str(val.get("unit") or unit or "").strip() or None
            key_str = f"{raw}|{u_str or ''}"
            stored_val: Any = dict(val)
        else:
            v_str = str(val).strip()
            if not v_str:
                return
            has_num = bool(re.search(r"\d", v_str))
            if not has_num and len(v_str.split()) > 3:
                return
            u_str = str(unit).strip() if unit is not None and str(unit).strip() else None
            key_str = f"{v_str}|{u_str or ''}"
            stored_val = val
        if key_str not in seen:
            seen.add(key_str)
            vals.append((stored_val, u_str))

    if corpus_register is not None:
        for ob in getattr(corpus_register, "obligations", ()) or ():
            params = getattr(ob, "parameters", None) or {}
            if "expected_value" in params:
                _add(params.get("expected_value"), params.get("expected_unit"))
            if "winner_value" in params:
                _add(params.get("winner_value"), params.get("winner_unit"))
            if "loser_value" in params:
                _add(params.get("loser_value"), params.get("loser_unit"))
            for cv in params.get("conflicting_values") or ():
                if isinstance(cv, Mapping):
                    _add(cv.get("value"), cv.get("unit"))
    for node in graph.claims.values():
        if getattr(node, "kind", "") == "opinion":
            continue
        _add(getattr(node, "value", None), getattr(node, "unit", None))
    return vals


def _audit_opinion_clause_factual_leakage(
    sub_cl: str,
    cl_nums: Sequence[Decimal],
    graph: JustificationGraph,
    *,
    vocabulary: Any = None,
    corpus_register: Any = None,
) -> dict[str, Any] | None:
    """Reject opinion-marked clauses that assert subordinate conflict values or Vocabulary predicate facts."""
    for gov_val, gov_unit, sub_val, sub_unit in _extract_register_conflict_pairs(corpus_register):
        if sub_val is not None and _val_matches_clause(sub_val, sub_cl, unit=sub_unit):
            if gov_val is None or not _val_matches_clause(gov_val, sub_cl, unit=gov_unit):
                return {
                    "code": "GOVERNING_VALUE_MISMATCH",
                    "status": "refuted",
                    "reason": (
                        f"Opinion clause '{sub_cl}' asserts subordinate/conflicting value '{sub_val}' "
                        f"instead of governing value '{gov_val}'."
                    ),
                    "severity": "error",
                    "clause": sub_cl,
                }

    norm_cl = re.sub(r"\s+", " ", sub_cl.strip().lower())
    for ps in (getattr(vocabulary, "predicates", ()) or () if vocabulary is not None else ()):
        pred_terms = (ps.name, ps.name.replace("_", " "), *(getattr(ps, "aliases", ()) or ()))
        if not any(_word_phrase_in_text(str(t), norm_cl) for t in pred_terms):
            continue
        has_enum_val = (
            ps.value.type in ("enum", "set")
            and any(_word_phrase_in_text(str(ev).replace("_", " "), norm_cl) for ev in ps.value.enum)
        )
        if cl_nums or has_enum_val:
            return {
                "code": "OPINION_CARRIES_FACT",
                "status": "refuted",
                "reason": (
                    f"Opinion clause '{sub_cl}' carries concrete factual value for vocabulary "
                    f"predicate '{ps.name}'; factual claims must use extraction/derivation leaves."
                ),
                "severity": "error",
                "clause": sub_cl,
                "predicate": ps.name,
            }

    for val_item, u_str in _collect_concrete_fact_values(graph, corpus_register=corpus_register):
        if _val_matches_clause(val_item, sub_cl, unit=u_str):
            return {
                "code": "OPINION_CARRIES_FACT",
                "status": "refuted",
                "reason": (
                    f"Opinion clause '{sub_cl}' states factual value '{val_item}' instead of confining "
                    "concrete values to verified extraction/derivation claims."
                ),
                "severity": "error",
                "clause": sub_cl,
            }
    return None


def _decompose_and_verify_clauses(
    clauses: list[str],
    graph: JustificationGraph,
    target_vals_to_match: list[Decimal],
    exact_lex: set[str],
    stem_lex: set[str],
    subj: str | None,
    graph_relations: list[GraphRelationTuple],
    verified_raw_equations: Sequence[str],
    *,
    unit_registry: UnitRegistry | None = None,
    has_material_conflict: bool,
    semantic_evaluator: Any = None,
    vocabulary: Any = None,
    corpus_register: Any = None,
) -> tuple[bool, list[ResponseClauseClaim], list[ResponseRelationTuple], list[dict[str, Any]]]:
    """Decompose text into clauses, check conclusion polarity inversion, relational tuples & ungrounded claims."""
    conc_negative = _is_graph_conclusion_negative(graph)
    conc = graph.claims[graph.conclusion_id]
    clause_claims: list[ResponseClauseClaim] = []
    all_relations: list[ResponseRelationTuple] = []
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

            # Marked interpretive opinions bound to a kind='opinion' claim with premises
            # are explicitly permitted and do not require factual lexical grounding.
            reg_issue = _check_register_status_contradiction(
                sub_cl,
                vocabulary=vocabulary,
                corpus_register=corpus_register,
            )
            leak_issue: dict[str, Any] | None = None
            is_bound_opinion = False
            if reg_issue is not None:
                is_grounded = False
                issues.append(reg_issue)
            elif _OPINION_MARKER_RE.search(sub_cl):
                leak_issue = _audit_opinion_clause_factual_leakage(
                    sub_cl,
                    cl_nums,
                    graph,
                    vocabulary=vocabulary,
                    corpus_register=corpus_register,
                )
                if leak_issue is not None:
                    is_grounded = False
                    issues.append(leak_issue)
                else:
                    op_ok, _ = _clause_binds_to_graph_claim(
                        sub_cl,
                        cl_nums,
                        graph,
                        role=role,
                        has_material_conflict=has_material_conflict,
                        doc_q_exact=exact_lex,
                    )
                    is_bound_opinion = op_ok

            # Flag ungrounded substantive non-numeric assertions (avoid noise on pure formulas/evasion)
            if dedup_ungrounded and role != "formula" and not is_bound_opinion and reg_issue is None and leak_issue is None:
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

            if not any(
                i.get("code") in (
                    "UNSUPPORTED_FACTUAL_ADDITION",
                    "REGISTER_STATUS_CONTRADICTED",
                    "GOVERNING_VALUE_MISMATCH",
                    "OPINION_CARRIES_FACT",
                )
                and i.get("clause") == sub_cl
                for i in issues
            ):
                bound_ok, bind_reason = _clause_binds_to_graph_claim(
                    sub_cl,
                    cl_nums,
                    graph,
                    role=role,
                    has_material_conflict=has_material_conflict,
                    doc_q_exact=exact_lex,
                )
                if not bound_ok:
                    is_grounded = False
                    issues.append({
                        "code": "UNSUPPORTED_FACTUAL_ADDITION",
                        "status": "refuted",
                        "reason": bind_reason or f"Clause '{sub_cl}' is not linked to any verified graph leaf.",
                        "severity": "error",
                        "clause": sub_cl,
                    })

            sub_rels: list[ResponseRelationTuple] = []
            if role != "formula" and not is_formula and graph_relations:
                sub_rels, rel_issues = _verify_subclause_relations(
                    sub_cl,
                    graph_relations,
                    subj,
                    verified_raw_equations,
                    unit_registry=unit_registry,
                    semantic_evaluator=semantic_evaluator,
                )
                all_relations.extend(sub_rels)
                issues.extend(rel_issues)

            clause_claims.append(ResponseClauseClaim(
                clause_text=sub_cl,
                role=role,
                subject=subj,
                polarity=polarity,
                numbers=cl_nums,
                is_grounded=is_grounded,
                ungrounded_tokens=dedup_ungrounded,
                relations=tuple(sub_rels),
            ))

    return conclusion_polarity_inverted, clause_claims, all_relations, issues


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
    vocabulary: Any = None,
    corpus_register: Any = None,
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
        if getattr(doc, "authority_rank", None) is not None:
            known_doc_nums.add(Decimal(doc.authority_rank))
        for m_rng in re.finditer(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", doc.text):
            lo_i, hi_i = int(m_rng.group(1)), int(m_rng.group(2))
            if 1 <= lo_i < hi_i <= 50 and (hi_i - lo_i) <= 10:
                for step_n in range(lo_i, hi_i + 1):
                    known_doc_nums.add(Decimal(step_n))
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
    # Add unit-equivalent numeric magnitudes for rate/spread nodes in graph (e.g. 0.25 pp <-> 25 bps <-> 0.0025)
    for node in anc_nodes + [conc]:
        nv = _to_dec(node.value)
        if nv is None or not node.unit:
            continue
        u_low = _norm_token(node.unit)
        if u_low in {"percentage_point", "percentage point", "pp", "percentage_points_per_annum", "percent", "percent_per_annum", "%"}:
            known_doc_nums.add(nv * Decimal(100))
            known_doc_nums.add(nv / Decimal(100))
        elif u_low in {"basis_point", "basis point", "bp", "bps", "basis points", "basis_points_per_annum"}:
            known_doc_nums.add(nv / Decimal(100))
            known_doc_nums.add(nv / Decimal(10000))

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
                doc_id_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", str(getattr(doc, "document_id", "") or ""))
                allowed_cap_words.update(w.lower() for w in doc_id_words)
                for m_d in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})\b", doc.text):
                    w_low = m_d.group(1).lower()
                    if w_low not in _COMMON_STOPWORDS and w_low not in subj_tokens and w_low not in unit_words_low:
                        other_doc_entities.add(w_low)
            for gnode in graph.claims.values():
                for src_txt in (
                    str(getattr(gnode, "subject", "") or ""),
                    str(getattr(gnode, "predicate", "") or "").replace("_", " "),
                    *(str(getattr(ev, "quote", "") or "") for ev in (getattr(gnode, "evidence", ()) or ())),
                ):
                    allowed_cap_words.update(w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", src_txt))
            allowed_cap_words.update(_EPISTEMIC_META_TOKENS)
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
        if not matched_graph_val and equation_checks:
            for tv in target_vals_to_match:
                if any(
                    ec.status == "verified"
                    and ec.rhs_value is not None
                    and abs(ec.rhs_value - tv) <= Decimal("0.01")
                    for ec in equation_checks
                ):
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
        graph_units = {
            _UNIT_WORD_MAP.get(_norm_token(n.unit), (n.unit, ""))[0]
            for n in (anc_nodes + [conc])
            if n.unit
        }
        has_rate_spread_context = bool(
            graph_units & {"percent", "percent_per_annum", "percentage_point", "percentage_points_per_annum", "basis_point", "basis_points_per_annum"}
        )

        def _unit_tok_re(tok: str) -> str:
            return rf"(?<!\w){re.escape(tok)}(?!\w)" if tok.isalpha() else re.escape(tok)

        target_unit_present = any(re.search(_unit_tok_re(s), low_text) for s in target_unit_syns)
        wrong_units_detected: list[str] = []
        for uname, syns in _UNIT_SYNONYMS.items():
            if uname == conc.unit:
                continue
            if {conc.unit, uname} <= {"USD", "USD_per_year"}:
                continue
            if {conc.unit, uname} <= {"percent", "percent_per_annum"}:
                continue
            if has_rate_spread_context and uname in {"percentage_point", "percentage_points_per_annum", "basis_point", "basis_points_per_annum", "percent", "percent_per_annum"}:
                continue
            if uname in graph_units:
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

    # 9. Claim-by-claim decomposition: conclusion polarity + relational tuples + non-numeric & numeric additions
    graph_relations = build_supported_graph_relations(
        graph,
        documents,
        include_conflict_alternatives=bool(has_material_conflict or conflict_disclosed),
    )
    verified_raw_eqs = [ec.raw_equation for ec in equation_checks if ec.status == "verified"]
    conclusion_polarity_inverted, clause_claims, relation_checks, clause_issues = _decompose_and_verify_clauses(
        clauses,
        graph,
        target_vals_to_match,
        exact_lex,
        stem_lex,
        subj,
        graph_relations,
        verified_raw_eqs,
        unit_registry=unit_registry,
        has_material_conflict=has_material_conflict,
        semantic_evaluator=semantic_evaluator,
        vocabulary=vocabulary,
        corpus_register=corpus_register,
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

    doc_and_q_text = " ".join(
        [d.text for d in (documents or {}).values()]
        + [str(getattr(d, "document_id", "") or "") for d in (documents or {}).values()]
        + ([q_text] if q_text else [])
    )
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

    # 10. Financial assumption disclosure, unqualified damages & percent-vs-pp confusion
    financial_assumptions_disclosed = bool(_FINANCIAL_ASSUMPTION_DISCLOSURE_RE.search(text))
    unqualified_damages_claim = False
    for cl in clauses:
        for m_dm in _UNQUALIFIED_DAMAGES_CLAIM_RE.finditer(cl):
            prefix_dm = cl[: m_dm.start()]
            if not (
                _NEGATED_LIMIT_CONTEXT_RE.search(prefix_dm)
                or re.search(r"\b(?:not|no|neither|nor|never|without|sin)\b[^.;,]{0,40}$", prefix_dm, re.IGNORECASE)
            ):
                unqualified_damages_claim = True
                break
        if unqualified_damages_claim:
            break
    if unqualified_damages_claim:
        issues.append({
            "code": "UNQUALIFIED_DAMAGES_OR_CASHFLOW_CLAIM",
            "status": "refuted",
            "reason": "Response presents an illustrative rate calculation as exact legal damages or guaranteed realized cashflow.",
            "severity": "error",
        })

    percent_vs_pp_confusion = bool(_RELATIVE_PCT_CONFUSION_RE.search(text))
    if conc.unit in {"percentage_point", "percentage_points_per_annum", "basis_point", "basis_points_per_annum"} and not percent_vs_pp_confusion:
        has_pp_or_bps_word = bool(
            re.search(
                r"\b(?:pp|bps?|percentage\s+points?|basis\s+points?|puntos?\s+porcentuales?|puntos?\s+b[aá]sicos?)\b",
                text,
                re.IGNORECASE,
            )
        )
        if conc_val is not None and not has_pp_or_bps_word:
            val_str = str(conc_val).rstrip("0").rstrip(".") if "." in str(conc_val) else str(conc_val)
            if re.search(rf"\b{re.escape(val_str)}\s*(?:%|percent\b|por\s+ciento)", text, re.IGNORECASE):
                percent_vs_pp_confusion = True
    if percent_vs_pp_confusion:
        issues.append({
            "code": "PERCENT_VS_PERCENTAGE_POINT_CONFUSION",
            "status": "refuted",
            "reason": "Response confuses percentage points / basis points with relative percent difference.",
            "severity": "error",
        })

    relational_mismatch_in_response = any(
        iss.get("code") == "RELATIONAL_VALUE_MISMATCH" for iss in issues
    )
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
        "relational_mismatch_in_response": relational_mismatch_in_response,
        "qualified_as_subtotal": qualified_as_subtotal,
        "presents_subtotal_as_total": presents_subtotal_as_total,
        "conflict_disclosed": conflict_disclosed,
        "financial_assumptions_disclosed": financial_assumptions_disclosed,
        "unqualified_damages_claim": unqualified_damages_claim,
        "percent_vs_pp_confusion": percent_vs_pp_confusion,
        "has_missing_precision_disclosure": bool(_MISSING_PRECISION_DISCLOSURE_RE.search(text)),
        "equation_checks": [ec.to_dict() for ec in equation_checks],
        "structured_conditions": [sc.to_dict() for sc in observed_conds],
        "supported_relations": [gr.to_dict() for gr in graph_relations],
        "relation_checks": [rel.to_dict() for rel in relation_checks],
        "clause_claims": [cc.to_dict() for cc in clause_claims],
        "issues": tuple(issues),
    }
