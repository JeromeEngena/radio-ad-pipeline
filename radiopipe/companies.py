"""Find company / brand names in a transcript.

Layers, cheapest first:
 1. exact match against every known name and alias (one compiled regex)
 2. fuzzy match of word n-grams against known names (catches ASR misspellings)
 3. discovery of NEW names: sponsor phrases, legal-suffix patterns, spaCy ORG
    entities (English), and optionally Claude. New names are stored unverified
    so a person can confirm them in company_names.csv / the DB.
"""
import json
import os
import re
import unicodedata

from rapidfuzz import fuzz, process

LEGAL_SUFFIXES = {"ltd", "limited", "plc", "inc", "co", "company", "smc", "llc"}
STOP_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "uganda", "kampala", "africa",
    "government", "ministry", "president", "call", "visit", "today", "now", "the",
    "radio", "fm", "ugx", "shillings",
}

SPONSOR_RE = re.compile(
    r"\b(?:brought to you by|sponsored by|presented by|courtesy of|"
    r"in partnership with|powered by)\s+"
    r"([A-Z][\w&'\-\.]*(?:\s+(?:&\s+)?[A-Z][\w&'\-\.]*){0,3})")
SUFFIX_RE = re.compile(
    r"\b((?:[A-Z][\w&'\-]*\s+){1,3}"
    r"(?:Ltd|Limited|PLC|Plc|Bank|Sacco|Insurance|Telecom|Hospital|University|"
    r"Foods|Motors|Group|Company|Cooperative))\b")


def normalize(name):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t]
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def load_seed(conn, path):
    """Load 'Name|alias|alias' lines into companies/aliases. Safe to re-run."""
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            key = normalize(parts[0])
            if not key:
                continue
            conn.execute(
                "INSERT INTO companies(key,name,mentions,verified,source) VALUES(?,?,0,1,'seed') "
                "ON CONFLICT(key) DO UPDATE SET verified=1", (key, parts[0]))
            for alias in parts[1:]:
                ak = normalize(alias)
                if ak and ak != key:
                    conn.execute("INSERT OR REPLACE INTO aliases VALUES(?,?)", (ak, key))
            n += 1
    conn.commit()
    return n


class CompanyExtractor:
    def __init__(self, cfg, conn):
        self.cfg = cfg
        self.conn = conn
        self.threshold = cfg["fuzzy_threshold"]
        self._sig = None
        self.names = {}          # company_key -> display name
        self.lookup = {}         # key or alias_key -> company_key
        self._regex = None
        self._keys = []
        self.nlp = None
        if cfg.get("use_spacy"):
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
            except Exception:
                self.nlp = None
        self.llm = None
        if cfg.get("use_llm") and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.llm = anthropic.Anthropic()
            except Exception:
                self.llm = None
        self.refresh()

    # -- known-name index -------------------------------------------------
    def refresh(self):
        c = self.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        a = self.conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0]
        if (c, a) == self._sig:
            return
        self._sig = (c, a)
        self.names = {r["key"]: r["name"] for r in self.conn.execute("SELECT key,name FROM companies")}
        self.lookup = {k: k for k in self.names}
        for r in self.conn.execute("SELECT alias_key, company_key FROM aliases"):
            self.lookup[r["alias_key"]] = r["company_key"]
        usable = sorted((k for k in self.lookup if len(k) >= 3), key=len, reverse=True)
        self._keys = usable
        self._regex = (re.compile(r"\b(" + "|".join(re.escape(k) for k in usable) + r")\b")
                       if usable else None)

    def top_names(self, n=40):
        rows = self.conn.execute(
            "SELECT name FROM companies ORDER BY mentions DESC LIMIT ?", (n,)).fetchall()
        return [r["name"] for r in rows]

    # -- extraction ---------------------------------------------------------
    def extract(self, text):
        self.refresh()
        found = {}  # company_key -> mention dict
        norm = normalize(text)
        covered = set()

        if self._regex:
            for m in self._regex.finditer(norm):
                ck = self.lookup[m.group(1)]
                found.setdefault(ck, dict(key=ck, name=self.names[ck], method="exact",
                                          score=100.0, surface=m.group(1)))
                covered.update(range(len(norm[:m.start()].split()),
                                     len(norm[:m.end()].split())))

        if self._keys:
            toks = norm.split()
            for n in (1, 2, 3, 4):
                for i in range(len(toks) - n + 1):
                    if covered.intersection(range(i, i + n)):
                        continue
                    gram = " ".join(toks[i:i + n])
                    if len(gram) < 6:
                        continue
                    hit = process.extractOne(gram, self._keys, scorer=fuzz.ratio,
                                             score_cutoff=self.threshold)
                    if hit:
                        ck = self.lookup[hit[0]]
                        found.setdefault(ck, dict(key=ck, name=self.names[ck], method="fuzzy",
                                                  score=float(hit[1]), surface=gram))
                        covered.update(range(i, i + n))

        for surface, method in self._discover(text):
            key = normalize(surface)
            if len(key) < 3 or key in STOP_NAMES or key.isdigit():
                continue
            if key in self.lookup:
                ck = self.lookup[key]
                found.setdefault(ck, dict(key=ck, name=self.names[ck], method=method,
                                          score=95.0, surface=surface))
                continue
            if self._keys:
                hit = process.extractOne(key, self._keys, scorer=fuzz.ratio,
                                         score_cutoff=self.threshold)
                if hit:
                    ck = self.lookup[hit[0]]
                    found.setdefault(ck, dict(key=ck, name=self.names[ck], method="fuzzy",
                                              score=float(hit[1]), surface=surface))
                    continue
            found.setdefault(key, dict(key=key, name=surface.strip(" .,'-"), method=method,
                                       score=80.0, surface=surface, new=True))
        return list(found.values())

    def _discover(self, text):
        out = []
        for rx, method in ((SPONSOR_RE, "pattern"), (SUFFIX_RE, "pattern")):
            for m in rx.finditer(text):
                out.append((m.group(1).strip(" .,'-"), method))
        if self.nlp is not None:
            for ent in self.nlp(text[:3000]).ents:
                if ent.label_ == "ORG":
                    out.append((ent.text.strip(), "ner"))
        if self.llm is not None and text.strip():
            out.extend((n, "llm") for n in self._llm_names(text))
        return out

    def _llm_names(self, text):
        try:
            msg = self.llm.messages.create(
                model=self.cfg["llm_model"], max_tokens=300,
                system=("You extract advertiser, brand and company names from Ugandan radio "
                        "advert transcripts. Reply with ONLY a JSON array of strings."),
                messages=[{"role": "user", "content": text[:4000]}])
            raw = msg.content[0].text.strip()
            raw = raw[raw.find("["): raw.rfind("]") + 1]
            return [str(x) for x in json.loads(raw)]
        except Exception:
            return []
