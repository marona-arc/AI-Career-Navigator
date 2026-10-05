import re

KNOWN_INDUSTRIES = [
    ("FinTech", [r"\bfintech\b", r"\bfinancial\s+technology\b"]),
    ("Finance", [r"\bfinance\b", r"\bfinancial\b", r"\bbanking\b", r"\baccounting\b", r"\baudit\b"]),
    ("Healthcare", [r"\bhealthcare\b", r"\bhealth\b", r"\bmedical\b", r"\bpharma\b", r"\bbiotech\b", r"\bclinical\b"]),
    ("E-Commerce", [r"\be-?commerce\b", r"\bonline\s+retail\b", r"\bmarketplace\b", r"\bshop\b"]),
    ("Gaming", [r"\bgaming\b", r"\bgames?\s+development\b", r"\bvideo\s+games?\b"]),
    ("Technology", [r"\btechnology\b", r"\bsoftware\b", r"\btech\b", r"\bit\s+services\b", r"\bcloud\b", r"\bcybersecurity\b"]),
    ("Retail", [r"\bretail\b", r"\bfashion\b", r"\bconsumer\s+goods\b", r"\bfmcg\b"]),
    ("Education", [r"\beducation\b", r"\bedtech\b", r"\blearning\b", r"\be-?learning\b", r"\buniversity\b"]),
    ("Automotive", [r"\bautomotive\b", r"\bvehicles?\b", r"\bmobility\b", r"\bcars?\b"]),
    ("Consulting", [r"\bconsulting\b", r"\badvisory\b", r"\bmanagement\s+consulting\b"]),
    ("Telecommunications", [r"\btelecommunications?\b", r"\btelecom\b", r"\bnetwork\s+provider\b"]),
    ("Manufacturing", [r"\bmanufacturing\b", r"\bindustrial\b", r"\bfactory\b", r"\bproduction\b"]),
    ("Media", [r"\bmedia\b", r"\bpublishing\b", r"\bjunket\b", r"\bbroadcasting\b", r"\bentertainment\b"]),
    ("Human Resources", [r"\bhuman\s+resources\b", r"\bhr\b", r"\brecruiting\b", r"\btalent\s+acquisition\b"]),
]

def extract_industry(text: str, tags: list[str] = None) -> str:
    combined = (str(text) + " " + " ".join(tags or [])).lower()
    for ind_name, patterns in KNOWN_INDUSTRIES:
        for p in patterns:
            if re.search(p, combined):
                return ind_name
    return "Unknown"

# Test samples
print("Software Engineering tag ->", extract_industry("", ["Software Engineering"]))
print("Finance tag ->", extract_industry("", ["Finance"]))
print("Healthcare text ->", extract_industry("Experienced clinical data analyst in a major hospital"))
print("Random text ->", extract_industry("Just an ordinary job"))
