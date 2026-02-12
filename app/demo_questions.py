"""
Demo Questions for Public PitchBook Observer
=============================================
~100 synthetic questions covering various query types for demo and testing.
"""

DEMO_QUESTIONS = [
    # =========================================================================
    # DEAL SIGNALS - Funding Rounds (25 questions)
    # =========================================================================
    {
        "id": 1,
        "category": "deals_funding",
        "question": "What funding rounds were announced this week?",
        "expected_intent": "deals"
    },
    {
        "id": 2,
        "category": "deals_funding",
        "question": "Show me recent Series A announcements",
        "expected_intent": "deals"
    },
    {
        "id": 3,
        "category": "deals_funding",
        "question": "Which companies raised seed funding recently?",
        "expected_intent": "deals"
    },
    {
        "id": 4,
        "category": "deals_funding",
        "question": "Any Series B or Series C rounds this week?",
        "expected_intent": "deals"
    },
    {
        "id": 5,
        "category": "deals_funding",
        "question": "What's the largest funding round announced recently?",
        "expected_intent": "deals"
    },
    {
        "id": 6,
        "category": "deals_funding",
        "question": "Tell me about venture capital investments this week",
        "expected_intent": "deals"
    },
    {
        "id": 7,
        "category": "deals_funding",
        "question": "Which startups received funding?",
        "expected_intent": "deals"
    },
    {
        "id": 8,
        "category": "deals_funding",
        "question": "Show funding announcements from press releases",
        "expected_intent": "deals"
    },
    {
        "id": 9,
        "category": "deals_funding",
        "question": "What companies announced investment rounds?",
        "expected_intent": "deals"
    },
    {
        "id": 10,
        "category": "deals_funding",
        "question": "Recent capital raises in the tech sector",
        "expected_intent": "deals"
    },
    {
        "id": 11,
        "category": "deals_funding",
        "question": "Show me Form D filings from this week",
        "expected_intent": "deals"
    },
    {
        "id": 12,
        "category": "deals_funding",
        "question": "What SEC filings indicate new funding?",
        "expected_intent": "deals"
    },
    {
        "id": 13,
        "category": "deals_funding",
        "question": "Growth stage funding announcements",
        "expected_intent": "deals"
    },
    {
        "id": 14,
        "category": "deals_funding",
        "question": "Late stage investment news",
        "expected_intent": "deals"
    },
    {
        "id": 15,
        "category": "deals_funding",
        "question": "Pre-seed and seed deals announced",
        "expected_intent": "deals"
    },
    {
        "id": 16,
        "category": "deals_funding",
        "question": "What fintech companies raised money?",
        "expected_intent": "deals"
    },
    {
        "id": 17,
        "category": "deals_funding",
        "question": "Healthcare startup funding news",
        "expected_intent": "deals"
    },
    {
        "id": 18,
        "category": "deals_funding",
        "question": "AI companies that received investment",
        "expected_intent": "deals"
    },
    {
        "id": 19,
        "category": "deals_funding",
        "question": "Climate tech funding announcements",
        "expected_intent": "deals"
    },
    {
        "id": 20,
        "category": "deals_funding",
        "question": "SaaS companies raising capital",
        "expected_intent": "deals"
    },
    {
        "id": 21,
        "category": "deals_funding",
        "question": "Enterprise software funding rounds",
        "expected_intent": "deals"
    },
    {
        "id": 22,
        "category": "deals_funding",
        "question": "Consumer tech investment news",
        "expected_intent": "deals"
    },
    {
        "id": 23,
        "category": "deals_funding",
        "question": "Biotech funding announcements this week",
        "expected_intent": "deals"
    },
    {
        "id": 24,
        "category": "deals_funding",
        "question": "Crypto and blockchain funding news",
        "expected_intent": "deals"
    },
    {
        "id": 25,
        "category": "deals_funding",
        "question": "What startups announced their first funding round?",
        "expected_intent": "deals"
    },

    # =========================================================================
    # DEAL SIGNALS - M&A (15 questions)
    # =========================================================================
    {
        "id": 26,
        "category": "deals_ma",
        "question": "What acquisitions were announced recently?",
        "expected_intent": "deals"
    },
    {
        "id": 27,
        "category": "deals_ma",
        "question": "Show me recent merger announcements",
        "expected_intent": "deals"
    },
    {
        "id": 28,
        "category": "deals_ma",
        "question": "Which companies were acquired this week?",
        "expected_intent": "deals"
    },
    {
        "id": 29,
        "category": "deals_ma",
        "question": "Tech acquisition news",
        "expected_intent": "deals"
    },
    {
        "id": 30,
        "category": "deals_ma",
        "question": "Startup exits and acquisitions",
        "expected_intent": "deals"
    },
    {
        "id": 31,
        "category": "deals_ma",
        "question": "Who is buying companies in the market?",
        "expected_intent": "deals"
    },
    {
        "id": 32,
        "category": "deals_ma",
        "question": "M&A activity this week",
        "expected_intent": "deals"
    },
    {
        "id": 33,
        "category": "deals_ma",
        "question": "Corporate acquisitions announced",
        "expected_intent": "deals"
    },
    {
        "id": 34,
        "category": "deals_ma",
        "question": "Strategic acquisitions in tech",
        "expected_intent": "deals"
    },
    {
        "id": 35,
        "category": "deals_ma",
        "question": "Private equity buyouts announced",
        "expected_intent": "deals"
    },
    {
        "id": 36,
        "category": "deals_ma",
        "question": "Which big companies made acquisitions?",
        "expected_intent": "deals"
    },
    {
        "id": 37,
        "category": "deals_ma",
        "question": "Startup acquisition targets this week",
        "expected_intent": "deals"
    },
    {
        "id": 38,
        "category": "deals_ma",
        "question": "IPO announcements and news",
        "expected_intent": "deals"
    },
    {
        "id": 39,
        "category": "deals_ma",
        "question": "Companies going public",
        "expected_intent": "deals"
    },
    {
        "id": 40,
        "category": "deals_ma",
        "question": "SPAC merger announcements",
        "expected_intent": "deals"
    },

    # =========================================================================
    # INVESTOR QUERIES (20 questions)
    # =========================================================================
    {
        "id": 41,
        "category": "investor",
        "question": "What are the most active investors this week?",
        "expected_intent": "investor"
    },
    {
        "id": 42,
        "category": "investor",
        "question": "Show me VC portfolio updates",
        "expected_intent": "investor"
    },
    {
        "id": 43,
        "category": "investor",
        "question": "Which venture capital firms made investments?",
        "expected_intent": "investor"
    },
    {
        "id": 44,
        "category": "investor",
        "question": "Andreessen Horowitz portfolio news",
        "expected_intent": "investor"
    },
    {
        "id": 45,
        "category": "investor",
        "question": "Sequoia investments this week",
        "expected_intent": "investor"
    },
    {
        "id": 46,
        "category": "investor",
        "question": "Y Combinator company updates",
        "expected_intent": "investor"
    },
    {
        "id": 47,
        "category": "investor",
        "question": "What did a16z invest in?",
        "expected_intent": "investor"
    },
    {
        "id": 48,
        "category": "investor",
        "question": "Private equity firm activity",
        "expected_intent": "investor"
    },
    {
        "id": 49,
        "category": "investor",
        "question": "New fund announcements",
        "expected_intent": "investor"
    },
    {
        "id": 50,
        "category": "investor",
        "question": "Which investors led funding rounds?",
        "expected_intent": "investor"
    },
    {
        "id": 51,
        "category": "investor",
        "question": "LP investments in venture funds",
        "expected_intent": "investor"
    },
    {
        "id": 52,
        "category": "investor",
        "question": "Corporate venture capital activity",
        "expected_intent": "investor"
    },
    {
        "id": 53,
        "category": "investor",
        "question": "Angel investor deals",
        "expected_intent": "investor"
    },
    {
        "id": 54,
        "category": "investor",
        "question": "Who invested in AI startups?",
        "expected_intent": "investor"
    },
    {
        "id": 55,
        "category": "investor",
        "question": "Fintech focused investors",
        "expected_intent": "investor"
    },
    {
        "id": 56,
        "category": "investor",
        "question": "Healthcare VC investments",
        "expected_intent": "investor"
    },
    {
        "id": 57,
        "category": "investor",
        "question": "Growth equity investments",
        "expected_intent": "investor"
    },
    {
        "id": 58,
        "category": "investor",
        "question": "Seed stage investors active this week",
        "expected_intent": "investor"
    },
    {
        "id": 59,
        "category": "investor",
        "question": "International investor activity in US startups",
        "expected_intent": "investor"
    },
    {
        "id": 60,
        "category": "investor",
        "question": "Portfolio company exits for VCs",
        "expected_intent": "investor"
    },

    # =========================================================================
    # COMPANY QUERIES (20 questions)
    # =========================================================================
    {
        "id": 61,
        "category": "company",
        "question": "Tell me about OpenAI",
        "expected_intent": "company"
    },
    {
        "id": 62,
        "category": "company",
        "question": "What does Stripe do?",
        "expected_intent": "company"
    },
    {
        "id": 63,
        "category": "company",
        "question": "Company profile for Anthropic",
        "expected_intent": "company"
    },
    {
        "id": 64,
        "category": "company",
        "question": "Latest news about startup XYZ",
        "expected_intent": "company"
    },
    {
        "id": 65,
        "category": "company",
        "question": "What is the company working on?",
        "expected_intent": "company"
    },
    {
        "id": 66,
        "category": "company",
        "question": "Company announcements this week",
        "expected_intent": "company"
    },
    {
        "id": 67,
        "category": "company",
        "question": "Startup press releases",
        "expected_intent": "company"
    },
    {
        "id": 68,
        "category": "company",
        "question": "Product launch announcements",
        "expected_intent": "company"
    },
    {
        "id": 69,
        "category": "company",
        "question": "New feature releases from tech companies",
        "expected_intent": "company"
    },
    {
        "id": 70,
        "category": "company",
        "question": "Company expansion news",
        "expected_intent": "company"
    },
    {
        "id": 71,
        "category": "company",
        "question": "Leadership changes at startups",
        "expected_intent": "company"
    },
    {
        "id": 72,
        "category": "company",
        "question": "New CEO appointments",
        "expected_intent": "company"
    },
    {
        "id": 73,
        "category": "company",
        "question": "Executive team changes",
        "expected_intent": "company"
    },
    {
        "id": 74,
        "category": "company",
        "question": "Company partnerships announced",
        "expected_intent": "company"
    },
    {
        "id": 75,
        "category": "company",
        "question": "Strategic partnerships in tech",
        "expected_intent": "company"
    },
    {
        "id": 76,
        "category": "company",
        "question": "Company milestones announced",
        "expected_intent": "company"
    },
    {
        "id": 77,
        "category": "company",
        "question": "Startup growth metrics shared",
        "expected_intent": "company"
    },
    {
        "id": 78,
        "category": "company",
        "question": "Company valuation news",
        "expected_intent": "company"
    },
    {
        "id": 79,
        "category": "company",
        "question": "Unicorn status announcements",
        "expected_intent": "company"
    },
    {
        "id": 80,
        "category": "company",
        "question": "Decacorn company news",
        "expected_intent": "company"
    },

    # =========================================================================
    # TREND / MARKET QUERIES (15 questions)
    # =========================================================================
    {
        "id": 81,
        "category": "trend",
        "question": "What's trending in the startup ecosystem?",
        "expected_intent": "trend"
    },
    {
        "id": 82,
        "category": "trend",
        "question": "Hot sectors for investment",
        "expected_intent": "trend"
    },
    {
        "id": 83,
        "category": "trend",
        "question": "What's happening in the market this week?",
        "expected_intent": "trend"
    },
    {
        "id": 84,
        "category": "trend",
        "question": "Emerging technology trends",
        "expected_intent": "trend"
    },
    {
        "id": 85,
        "category": "trend",
        "question": "Popular investment themes",
        "expected_intent": "trend"
    },
    {
        "id": 86,
        "category": "trend",
        "question": "Market sentiment this week",
        "expected_intent": "trend"
    },
    {
        "id": 87,
        "category": "trend",
        "question": "Which sectors are getting funded?",
        "expected_intent": "trend"
    },
    {
        "id": 88,
        "category": "trend",
        "question": "Startup ecosystem overview",
        "expected_intent": "trend"
    },
    {
        "id": 89,
        "category": "trend",
        "question": "VC market activity summary",
        "expected_intent": "trend"
    },
    {
        "id": 90,
        "category": "trend",
        "question": "What industries are growing?",
        "expected_intent": "trend"
    },
    {
        "id": 91,
        "category": "trend",
        "question": "Tech industry news summary",
        "expected_intent": "trend"
    },
    {
        "id": 92,
        "category": "trend",
        "question": "Weekly funding summary",
        "expected_intent": "trend"
    },
    {
        "id": 93,
        "category": "trend",
        "question": "Deal flow this week",
        "expected_intent": "trend"
    },
    {
        "id": 94,
        "category": "trend",
        "question": "Investment landscape overview",
        "expected_intent": "trend"
    },
    {
        "id": 95,
        "category": "trend",
        "question": "Macro trends affecting startups",
        "expected_intent": "trend"
    },

    # =========================================================================
    # GENERAL / EDGE CASE QUERIES (10 questions)
    # =========================================================================
    {
        "id": 96,
        "category": "general",
        "question": "Show me everything from this week",
        "expected_intent": "general"
    },
    {
        "id": 97,
        "category": "general",
        "question": "What's new?",
        "expected_intent": "general"
    },
    {
        "id": 98,
        "category": "general",
        "question": "Give me a summary of recent news",
        "expected_intent": "general"
    },
    {
        "id": 99,
        "category": "general",
        "question": "Any interesting announcements?",
        "expected_intent": "general"
    },
    {
        "id": 100,
        "category": "general",
        "question": "What should I know about this week?",
        "expected_intent": "general"
    },
    {
        "id": 101,
        "category": "general",
        "question": "Highlight the most important news",
        "expected_intent": "general"
    },
    {
        "id": 102,
        "category": "general",
        "question": "Breaking news in tech and startups",
        "expected_intent": "general"
    },
    {
        "id": 103,
        "category": "general",
        "question": "Top stories this week",
        "expected_intent": "general"
    },
    {
        "id": 104,
        "category": "general",
        "question": "What are people talking about?",
        "expected_intent": "general"
    },
    {
        "id": 105,
        "category": "general",
        "question": "News I might have missed",
        "expected_intent": "general"
    },
]


def get_questions_by_category(category: str = None):
    """Get questions, optionally filtered by category."""
    if category:
        return [q for q in DEMO_QUESTIONS if q["category"] == category]
    return DEMO_QUESTIONS


def get_random_questions(n: int = 10):
    """Get n random questions."""
    import random
    return random.sample(DEMO_QUESTIONS, min(n, len(DEMO_QUESTIONS)))


def get_question_categories():
    """Get list of unique categories."""
    return list(set(q["category"] for q in DEMO_QUESTIONS))


if __name__ == "__main__":
    print(f"Total demo questions: {len(DEMO_QUESTIONS)}")
    print(f"\nCategories:")
    for cat in get_question_categories():
        count = len(get_questions_by_category(cat))
        print(f"  {cat}: {count} questions")
