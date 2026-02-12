"""
Inject Demo Data
================
Creates synthetic sample documents for demo/testing purposes.
Generates ~50 realistic articles covering funding, M&A, investor activity, etc.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.database import init_database, upsert_artifact, insert_chunks, insert_evaluation
from pipeline.index import get_embedding_provider, VectorStore, KeywordIndex
from pipeline.config import load_pipeline_config, INDEXES_DIR, DATA_DIR, ensure_week_dirs

# Synthetic data templates
COMPANIES = [
    ("NeuralPath AI", "AI/ML", "San Francisco"),
    ("QuantumLeap Computing", "Quantum", "Boston"),
    ("GreenGrid Energy", "CleanTech", "Austin"),
    ("MedSecure Health", "Healthcare", "New York"),
    ("FinFlow Systems", "Fintech", "Chicago"),
    ("CloudNine Infrastructure", "Cloud", "Seattle"),
    ("DataVault Analytics", "Data", "Denver"),
    ("RoboLogix Automation", "Robotics", "Pittsburgh"),
    ("BioGenesis Labs", "Biotech", "San Diego"),
    ("CyberShield Security", "Cybersecurity", "Washington DC"),
    ("SpaceLink Satellites", "Space", "Los Angeles"),
    ("AgriTech Solutions", "AgTech", "Des Moines"),
    ("EduLearn Platform", "EdTech", "Atlanta"),
    ("PropTech Homes", "Real Estate", "Miami"),
    ("LogiChain Supply", "Logistics", "Dallas"),
]

INVESTORS = [
    ("Sequoia Capital", "VC"),
    ("Andreessen Horowitz", "VC"),
    ("Benchmark", "VC"),
    ("Accel Partners", "VC"),
    ("Lightspeed Venture Partners", "VC"),
    ("General Catalyst", "VC"),
    ("Index Ventures", "VC"),
    ("Founders Fund", "VC"),
    ("Greylock Partners", "VC"),
    ("NEA", "VC"),
    ("Tiger Global", "Growth"),
    ("SoftBank Vision Fund", "Growth"),
    ("Insight Partners", "Growth"),
    ("Silver Lake", "PE"),
    ("KKR", "PE"),
]

FUNDING_TEMPLATES = [
    "{company} Raises ${amount}M Series {series} to Expand {focus}",
    "{company} Announces ${amount} Million {series} Round Led by {investor}",
    "{company} Secures ${amount}M in Series {series} Funding",
    "{investor} Leads ${amount}M Investment in {company}",
    "{company} Closes ${amount} Million Series {series} to Accelerate Growth",
]

FUNDING_BODY_TEMPLATE = """
{company}, a {location}-based {sector} startup, today announced it has raised ${amount} million in Series {series} funding.

The round was led by {lead_investor}, with participation from {other_investors}. This brings the company's total funding to ${total}M.

"{quote}" said {ceo_name}, CEO and co-founder of {company}.

The funding will be used to {use_of_funds}.

{company} has grown {growth_metric} over the past year and now serves {customers} customers across {markets}.

About {company}:
{company} is transforming the {sector} industry with its innovative approach to {focus}. Founded in {founded_year}, the company is headquartered in {location}.

For more information, visit www.{company_domain}.com

Media Contact:
press@{company_domain}.com
"""

ACQUISITION_TEMPLATES = [
    "{acquirer} Acquires {target} for ${amount}M",
    "{acquirer} to Acquire {target} in ${amount} Million Deal",
    "{target} Acquired by {acquirer} to Strengthen {focus}",
    "{acquirer} Announces Acquisition of {target}",
]

ACQUISITION_BODY_TEMPLATE = """
{acquirer} today announced it has entered into a definitive agreement to acquire {target}, a leading provider of {target_focus} solutions.

The transaction, valued at approximately ${amount} million, is expected to close in {close_timeline}.

"{quote}" said {acquirer_ceo}, CEO of {acquirer}.

{target}, founded in {target_founded}, has built a strong reputation for {target_strength}. The company's {target_product} serves over {target_customers} customers.

The acquisition will enable {acquirer} to {synergy_description}.

All {target_employees} {target} employees are expected to join {acquirer}.

About {acquirer}:
{acquirer_description}

About {target}:
{target_description}
"""

INVESTOR_PORTFOLIO_TEMPLATE = """
{investor} Portfolio Update - {date}

Recent Investments:

{investments}

Portfolio Highlights:
- Total portfolio companies: {total_companies}
- New investments this quarter: {new_investments}
- Follow-on investments: {followons}

Notable exits this year:
{exits}

{investor} continues to focus on {focus_areas}.

For partnership inquiries: partnerships@{investor_domain}.com
"""

COMPANY_NEWS_TEMPLATE = """
{company} Announces {announcement}

{location} - {date} - {company}, a leading {sector} company, today announced {announcement_detail}.

Key highlights:
{highlights}

"{quote}" said {spokesperson}, {title} at {company}.

{additional_context}

About {company}:
{company} is {company_description}. Learn more at www.{company_domain}.com.
"""

def generate_funding_article(idx: int, week: str) -> dict:
    """Generate a synthetic funding announcement."""
    company_data = random.choice(COMPANIES)
    company, sector, location = company_data

    series = random.choice(["A", "A", "B", "B", "C", "Seed", "Seed"])
    amount = random.choice([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150])

    lead_investor = random.choice(INVESTORS)
    other_investors = random.sample([i for i in INVESTORS if i != lead_investor], k=min(3, len(INVESTORS)-1))

    title = random.choice(FUNDING_TEMPLATES).format(
        company=company,
        amount=amount,
        series=series,
        investor=lead_investor[0],
        focus=sector
    )

    body = FUNDING_BODY_TEMPLATE.format(
        company=company,
        location=location,
        sector=sector,
        amount=amount,
        series=series,
        lead_investor=lead_investor[0],
        other_investors=", ".join([i[0] for i in other_investors]),
        total=amount + random.randint(5, 50),
        quote=f"This funding will accelerate our mission to transform {sector}.",
        ceo_name=f"Jane Smith",
        use_of_funds=f"expand the engineering team, enter new markets, and accelerate product development",
        growth_metric=f"{random.randint(150, 400)}%",
        customers=random.randint(50, 500),
        markets=random.randint(3, 20),
        focus=sector.lower(),
        founded_year=random.randint(2018, 2023),
        company_domain=company.lower().replace(" ", "")
    )

    return {
        "title": title,
        "body": body,
        "source_kind": random.choice(["pr_wire", "news"]),
        "source_name": random.choice(["PR Newswire", "GlobeNewswire", "TechCrunch", "Business Wire"]),
        "url": f"https://example.com/news/funding-{idx}",
        "keywords": ["funding", "raises", "Series " + series, "investment", "venture", "capital"],
        "bucket": "deal_signal"
    }


def generate_acquisition_article(idx: int, week: str) -> dict:
    """Generate a synthetic acquisition announcement."""
    acquirer = random.choice(["Microsoft", "Google", "Amazon", "Salesforce", "Oracle", "IBM", "Cisco", "Adobe"])
    target_data = random.choice(COMPANIES)
    target, sector, location = target_data

    amount = random.choice([50, 100, 150, 200, 300, 500, 750, 1000, 1500])

    title = random.choice(ACQUISITION_TEMPLATES).format(
        acquirer=acquirer,
        target=target,
        amount=amount,
        focus=sector
    )

    body = ACQUISITION_BODY_TEMPLATE.format(
        acquirer=acquirer,
        target=target,
        target_focus=sector.lower(),
        amount=amount,
        close_timeline="Q" + str(random.randint(1,4)) + " 2026",
        quote=f"This acquisition strengthens our position in {sector}.",
        acquirer_ceo="John Doe",
        target_founded=random.randint(2015, 2022),
        target_strength=f"innovative {sector.lower()} solutions",
        target_product="platform",
        target_customers=random.randint(100, 1000),
        synergy_description=f"expand its {sector.lower()} capabilities",
        target_employees=random.randint(50, 300),
        acquirer_description=f"{acquirer} is a leading technology company.",
        target_description=f"{target} provides {sector.lower()} solutions."
    )

    return {
        "title": title,
        "body": body,
        "source_kind": "news",
        "source_name": random.choice(["Reuters", "Bloomberg", "TechCrunch", "Wall Street Journal"]),
        "url": f"https://example.com/news/acquisition-{idx}",
        "keywords": ["acquisition", "acquires", "acquired", "merger", "deal"],
        "bucket": "deal_signal"
    }


def generate_investor_update(idx: int, week: str) -> dict:
    """Generate a synthetic investor portfolio update."""
    investor_data = random.choice(INVESTORS)
    investor, inv_type = investor_data

    investments = []
    for _ in range(random.randint(3, 6)):
        company_data = random.choice(COMPANIES)
        amount = random.choice([5, 10, 15, 20, 30])
        investments.append(f"- {company_data[0]}: ${amount}M Series {random.choice(['A', 'B', 'Seed'])}")

    exits = []
    for _ in range(random.randint(1, 3)):
        company_data = random.choice(COMPANIES)
        exits.append(f"- {company_data[0]}: Acquired for ${random.randint(100, 500)}M")

    title = f"{investor} Portfolio Companies and Recent Investments"

    body = INVESTOR_PORTFOLIO_TEMPLATE.format(
        investor=investor,
        date=datetime.now().strftime("%B %Y"),
        investments="\n".join(investments),
        total_companies=random.randint(50, 200),
        new_investments=random.randint(5, 15),
        followons=random.randint(10, 30),
        exits="\n".join(exits),
        focus_areas="AI/ML, fintech, healthcare, and enterprise software",
        investor_domain=investor.lower().replace(" ", "")
    )

    return {
        "title": title,
        "body": body,
        "source_kind": "investor_portfolio",
        "source_name": investor,
        "url": f"https://{investor.lower().replace(' ', '')}.com/portfolio",
        "keywords": ["portfolio", "invested", "fund", "venture capital"],
        "bucket": "investor_graph_change"
    }


def generate_company_news(idx: int, week: str) -> dict:
    """Generate a synthetic company announcement."""
    company_data = random.choice(COMPANIES)
    company, sector, location = company_data

    announcement_types = [
        ("New Product Launch", f"the launch of its next-generation {sector.lower()} platform"),
        ("Expansion", f"expansion into {random.choice(['Europe', 'Asia Pacific', 'Latin America'])}"),
        ("Partnership", f"a strategic partnership with {random.choice(['Microsoft', 'Google', 'AWS', 'Salesforce'])}"),
        ("Milestone", f"reaching {random.randint(100, 1000)} enterprise customers"),
        ("Leadership", f"the appointment of {random.choice(['Sarah Johnson', 'Michael Chen', 'Emily Brown'])} as {random.choice(['CTO', 'CFO', 'COO'])}"),
    ]

    ann_type, ann_detail = random.choice(announcement_types)

    title = f"{company} Announces {ann_type}"

    body = COMPANY_NEWS_TEMPLATE.format(
        company=company,
        announcement=ann_type.lower(),
        location=location,
        date=datetime.now().strftime("%B %d, %Y"),
        sector=sector.lower(),
        announcement_detail=ann_detail,
        highlights="- Significant market expansion\n- Enhanced product capabilities\n- Strengthened leadership team",
        quote=f"This marks an exciting milestone for {company}.",
        spokesperson="Alex Rivera",
        title="VP of Communications",
        additional_context=f"{company} continues to lead innovation in the {sector.lower()} space.",
        company_description=f"a leading {sector.lower()} company",
        company_domain=company.lower().replace(" ", "")
    )

    return {
        "title": title,
        "body": body,
        "source_kind": "company_press",
        "source_name": f"{company} Blog",
        "url": f"https://{company.lower().replace(' ', '')}.com/news/{idx}",
        "keywords": ["announces", "launch", "expansion", "partnership", "leadership"],
        "bucket": "company_profile_change"
    }


def generate_sec_filing(idx: int, week: str) -> dict:
    """Generate a synthetic SEC Form D filing summary."""
    company_data = random.choice(COMPANIES)
    company, sector, location = company_data

    amount = random.choice([1000000, 2500000, 5000000, 10000000, 15000000, 25000000, 50000000])

    title = f"SEC Form D Filing: {company}"

    body = f"""
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549

FORM D
Notice of Exempt Offering of Securities

Name of Issuer: {company}
Street Address: 123 Tech Street
City: {location}
State: CA
Zip: 94105

Type of Securities: Equity
Total Offering Amount: ${amount:,}
Total Amount Sold: ${amount:,}
Total Remaining: $0

Investors:
- Number of Accredited Investors: {random.randint(5, 25)}
- Number of Non-Accredited Investors: 0

Sales Commissions: ${int(amount * 0.02):,}
Finders' Fees: ${int(amount * 0.01):,}

Use of Proceeds: Working capital, product development, and market expansion.

This filing indicates that {company} has completed a private placement of securities
exempt from registration under Rule 506(b) of Regulation D.

The offering was conducted to raise capital for the company's continued growth in the
{sector.lower()} market.
"""

    return {
        "title": title,
        "body": body,
        "source_kind": "filing",
        "source_name": "SEC EDGAR",
        "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={company.replace(' ', '+')}&type=D",
        "keywords": ["SEC", "Form D", "filing", "securities", "offering", "investment"],
        "bucket": "deal_signal"
    }


def inject_demo_data():
    """Main function to inject demo data."""
    print("\n" + "="*60)
    print("  Injecting Demo Data")
    print("="*60 + "\n")

    # Initialize
    print("[1/5] Initializing database...")
    init_database()

    config = load_pipeline_config()
    week = datetime.utcnow().strftime("%Y-W%W")
    dirs = ensure_week_dirs(week)

    print(f"[2/5] Generating synthetic articles for week {week}...")

    # Generate articles
    articles = []

    # Funding announcements (20)
    for i in range(20):
        articles.append(generate_funding_article(i, week))

    # Acquisitions (8)
    for i in range(8):
        articles.append(generate_acquisition_article(i, week))

    # Investor updates (7)
    for i in range(7):
        articles.append(generate_investor_update(i, week))

    # Company news (10)
    for i in range(10):
        articles.append(generate_company_news(i, week))

    # SEC filings (5)
    for i in range(5):
        articles.append(generate_sec_filing(i, week))

    print(f"    Generated {len(articles)} articles")

    print("[3/5] Storing artifacts and chunks...")

    all_chunks = []

    for i, article in enumerate(articles):
        # Create artifact
        content = f"# {article['title']}\n\n{article['body']}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        artifact_id = hashlib.sha256(f"{article['url']}::{content_hash}".encode()).hexdigest()[:16]

        artifact = {
            "artifact_id": artifact_id,
            "canonical_url": article["url"],
            "content_hash": content_hash,
            "source_name": article["source_name"],
            "source_kind": article["source_kind"],
            "domain": article["url"].split("/")[2],
            "title": article["title"],
            "url": article["url"],
            "week": week,
            "retrieved_at": datetime.utcnow().isoformat(),
            "published_at": (datetime.utcnow() - timedelta(days=random.randint(0, 6))).isoformat(),
            "fetch_mode": "http",
            "http_status": 200,
            "main_text_length": len(content)
        }

        upsert_artifact(artifact)

        # Create chunks (simple chunking)
        chunk_size = 1500
        text = content
        chunk_idx = 0

        while text:
            chunk_text = text[:chunk_size]
            text = text[chunk_size-200:] if len(text) > chunk_size else ""

            chunk_id = f"{artifact_id}_{chunk_idx:04d}"

            chunk = {
                "chunk_id": chunk_id,
                "artifact_id": artifact_id,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "start_char": chunk_idx * (chunk_size - 200),
                "end_char": chunk_idx * (chunk_size - 200) + len(chunk_text),
                "token_count_approx": len(chunk_text) // 4,
                "week": week,
                "source_kind": article["source_kind"],
                "source_name": article["source_name"],
                "canonical_url": article["url"],
                "title": article["title"],
                "published_at": artifact["published_at"],
                "retrieved_at": artifact["retrieved_at"]
            }

            all_chunks.append(chunk)
            chunk_idx += 1

            if chunk_idx > 3:  # Max 4 chunks per article
                break

        # Store evaluation
        insert_evaluation(
            artifact_id=artifact_id,
            week=week,
            bucket=article["bucket"],
            confidence=0.85,
            rationale=f"Matched keywords: {', '.join(article['keywords'][:3])}",
            keywords_matched=article["keywords"]
        )

    # Insert chunks to database
    insert_chunks(all_chunks)
    print(f"    Stored {len(articles)} artifacts, {len(all_chunks)} chunks")

    print("[4/5] Building vector index...")

    # Build index
    embed_cfg = config.index.get("embeddings", {})
    provider = get_embedding_provider(embed_cfg)

    persist_dir = INDEXES_DIR / "chroma"
    vector_store = VectorStore(persist_dir, provider.dimension)
    keyword_index = KeywordIndex(persist_dir)

    # Embed and store
    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "artifact_id": c["artifact_id"],
            "canonical_url": c["canonical_url"],
            "title": c["title"],
            "source_kind": c["source_kind"],
            "source_name": c["source_name"],
            "week": c["week"],
            "published_at": c["published_at"],
            "retrieved_at": c["retrieved_at"],
            "text": c["text"][:500]
        }
        for c in all_chunks
    ]

    print("    Generating embeddings...")
    embeddings = provider.embed(texts)

    print("    Adding to vector store...")
    vector_store.add(embeddings, metadatas)

    print("    Building keyword index...")
    for chunk in all_chunks:
        keyword_index.add(chunk["chunk_id"], chunk["text"])
    keyword_index.save()

    print("[5/5] Creating evaluation report...")

    # Create eval report for pargv_batch
    eval_report = {
        "week": week,
        "total_evaluated": len(articles),
        "buckets": {
            "deal_signal": {
                "count": len([a for a in articles if a["bucket"] == "deal_signal"]),
                "artifacts": [
                    {
                        "artifact_id": hashlib.sha256(f"{a['url']}::{hashlib.sha256((a['title']+a['body']).encode()).hexdigest()[:16]}".encode()).hexdigest()[:16],
                        "canonical_url": a["url"],
                        "source_name": a["source_name"],
                        "source_kind": a["source_kind"],
                        "title": a["title"],
                        "keywords_matched": a["keywords"]
                    }
                    for a in articles if a["bucket"] == "deal_signal"
                ]
            },
            "investor_graph_change": {
                "count": len([a for a in articles if a["bucket"] == "investor_graph_change"]),
                "artifacts": [
                    {
                        "artifact_id": hashlib.sha256(f"{a['url']}::{hashlib.sha256((a['title']+a['body']).encode()).hexdigest()[:16]}".encode()).hexdigest()[:16],
                        "canonical_url": a["url"],
                        "source_name": a["source_name"],
                        "source_kind": a["source_kind"],
                        "title": a["title"],
                        "keywords_matched": a["keywords"]
                    }
                    for a in articles if a["bucket"] == "investor_graph_change"
                ]
            },
            "company_profile_change": {
                "count": len([a for a in articles if a["bucket"] == "company_profile_change"]),
                "artifacts": [
                    {
                        "artifact_id": hashlib.sha256(f"{a['url']}::{hashlib.sha256((a['title']+a['body']).encode()).hexdigest()[:16]}".encode()).hexdigest()[:16],
                        "canonical_url": a["url"],
                        "source_name": a["source_name"],
                        "source_kind": a["source_kind"],
                        "title": a["title"],
                        "keywords_matched": a["keywords"]
                    }
                    for a in articles if a["bucket"] == "company_profile_change"
                ]
            },
            "noise": {"count": 0, "artifacts": []}
        },
        "evaluated_at": datetime.utcnow().isoformat()
    }

    eval_path = dirs["runs"] / "eval_report.json"
    eval_path.write_text(json.dumps(eval_report, indent=2))

    print("\n" + "="*60)
    print("  Demo Data Injection Complete!")
    print("="*60)
    print(f"""
Summary:
  - Articles: {len(articles)}
  - Chunks: {len(all_chunks)}
  - Vectors indexed: {vector_store.count()}
  - Week: {week}

Categories:
  - Deal signals (funding/M&A): {len([a for a in articles if a['bucket'] == 'deal_signal'])}
  - Investor updates: {len([a for a in articles if a['bucket'] == 'investor_graph_change'])}
  - Company news: {len([a for a in articles if a['bucket'] == 'company_profile_change'])}

Next steps:
  1. Run 'run_chat.bat' for interactive chat
  2. Run 'run_batch.bat' to test all demo questions
  3. Try queries like:
     - "What funding rounds were announced?"
     - "Tell me about acquisitions"
     - "Which investors are active?"
""")


if __name__ == "__main__":
    inject_demo_data()
