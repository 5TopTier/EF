import os
import requests
import xml.etree.ElementTree as ET

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN:
    raise ValueError("Missing DISCORD_TOKEN in .env")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

citation_store = {}

SYSTEM_PROMPT = """
IDENTITY
You are the Euphoria Fit Research Bot. Your purpose is to answer
health and fitness questions by synthesizing the best available
peer-reviewed evidence — not the opinion of any single organization,
trainer, or methodology.

You have been provided with abstracts from PubMed retrieved in real
time for the user's question. Base your response on these abstracts.
Do not make claims that go beyond what the retrieved research supports.

──────────────────────────────────────────
RESPONSE FORMAT
──────────────────────────────────────────

Every response must follow this structure:

1. WHAT THE RESEARCH SHOWS
   State the main finding from the best available evidence.
   Lead with the highest quality evidence level present.

2. EVIDENCE QUALITY
   State the level: 'This is based on [X] — [meta-analysis /
   RCTs / cohort studies / limited evidence].'
   Be honest about weak or limited evidence.

3. CONFLICTING FINDINGS (include ONLY if conflict exists)
   'Some research shows a different picture:'
   Describe the conflicting finding.
   Explain WHY it may conflict: different population, shorter
   duration, different methodology, different outcome measured.
   State what the weight of evidence suggests overall.

4. PRACTICAL TAKEAWAY
   One to three sentences. What does this mean for training,
   nutrition, or recovery in practice?

5. CITATION OFFER
   Always end with: 'Want the full citations? Type !cite'

──────────────────────────────────────────
EVIDENCE RULES
──────────────────────────────────────────

- Prioritize meta-analyses and systematic reviews above all else
- Prefer RCTs over observational studies
- If only weak evidence exists, say so explicitly
- Never present a single study as definitive
- Acknowledge when a finding is preliminary or under active debate
- When research is sparse, say: 'Evidence here is limited.'
  Do not fill gaps with speculation.

──────────────────────────────────────────
CONFLICT HANDLING
──────────────────────────────────────────

When studies conflict, you MUST:
a) Present both findings fairly — do not choose a side
b) Explain the most likely reason for the conflict:
   - Population differences (trained vs untrained, male vs female,
     older vs younger)
   - Study duration (short-term vs long-term outcomes differ)
   - Methodology (lab conditions vs real-world)
   - Outcome measured (performance vs body composition vs health)
   - Dosage or protocol differences
c) State what the majority or highest-quality evidence suggests
d) Use language like: 'The current weight of evidence favors X,
   though this remains an area of active research.'

──────────────────────────────────────────
ABSOLUTE LIMITS
──────────────────────────────────────────

NEVER do any of the following:

- Diagnose injuries, conditions, or symptoms
- Recommend specific medications or therapeutic doses
- Validate pseudoscientific claims (detox, spot reduction,
  muscle toning, alkaline diet, metabolic confusion)
  Instead: explain what the research actually shows
- Present organizational guidelines (ACSM, WHO, etc.) as
  equivalent to primary research — cite them as context only
- Make claims that go beyond the retrieved abstracts
- Use hedging language to hide uncertainty — name it directly

For medical questions: 'For this I recommend consulting a
physician or physical therapist who can assess you directly.'

──────────────────────────────────────────
TONE
──────────────────────────────────────────

- Speak like the most knowledgeable person in the room who
  genuinely wants you to understand — not to impress you
- Plain language. Define technical terms when necessary.
- Intellectual honesty over confidence. Uncertainty is not
  weakness — stating it clearly is the mark of rigor.
- No hype. No transformation language. No superlatives.
- Responses should be thorough but not exhaustive.
  Aim for 150-300 words per response.
""".strip()


def search_pubmed(query, max_results=8):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    enhanced_query = f"{query} AND (systematic[sb] OR meta-analysis[pt] OR randomized controlled trial[pt])"

    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": enhanced_query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
        "datetype": "pdat",
        "reldate": 1825,
    }

    try:
        search_resp = requests.get(search_url, params=search_params, timeout=20)
        search_resp.raise_for_status()
        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if len(pmids) < 3:
            search_params["term"] = query
            search_resp = requests.get(search_url, params=search_params, timeout=20)
            search_resp.raise_for_status()
            pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return [], []

        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }

        fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=20)
        fetch_resp.raise_for_status()
        root = ET.fromstring(fetch_resp.content)

        abstracts = []
        citations = []

        for article in root.findall(".//PubmedArticle"):
            abstract_texts = article.findall(".//AbstractText")
            abstract_parts = []
            for t in abstract_texts:
                text = "".join(t.itertext()).strip()
                if text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else "Unknown Title"

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text.strip() if journal_el is not None and journal_el.text else "Unknown Journal"

            year = "Unknown Year"
            year_el = article.find(".//PubDate/Year")
            medline_date_el = article.find(".//PubDate/MedlineDate")
            if year_el is not None and year_el.text:
                year = year_el.text.strip()
            elif medline_date_el is not None and medline_date_el.text:
                year = medline_date_el.text.strip()

            authors = article.findall(".//Author")
            author_names = []
            for author in authors[:3]:
                last = author.find("LastName")
                if last is not None and last.text:
                    author_names.append(last.text.strip())

            author_str = ", ".join(author_names) if author_names else "Unknown authors"
            if len(authors) > 3 and author_names:
                author_str += " et al."

            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
            pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            if abstract:
                abstracts.append(
                    f"STUDY: {title}\n"
                    f"JOURNAL: {journal} ({year})\n"
                    f"FINDINGS: {abstract[:1200]}"
                )
                citations.append(f'{author_str}. "{title}" {journal}. {year}. {pubmed_link}')

        return abstracts, citations

    except Exception:
        return [], []


def synthesize_with_ai(question, abstracts):
    if not abstracts:
        research_context = "No relevant PubMed abstracts were found for this question."
    else:
        research_context = "\n\n".join(abstracts)

    user_message = f"""
USER QUESTION:
{question}

RETRIEVED PUBMED ABSTRACTS:
{research_context}

Based on these abstracts, answer the user's question following the required response format exactly.
""".strip()

    response = client.responses.create(
        model="gpt-5.2",
        instructions=SYSTEM_PROMPT,
        input=user_message,
    )

    return (response.output_text or "").strip()


async def send_long_message(channel, text, limit=1900):
    if not text:
        await channel.send("I could not generate a response.")
        return

    chunks = [text[i:i + limit] for i in range(0, len(text), limit)]
    for chunk in chunks:
        await channel.send(chunk)


@bot.event
async def on_ready():
    print(f"Euphoria Fit Research Bot is online as {bot.user}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    if message.content.startswith("!"):
        await bot.process_commands(message)
        return

    if message.channel.name != "ask-the-science":
        await bot.process_commands(message)
        return

    if len(message.content.strip()) < 10:
        await bot.process_commands(message)
        return

    async with message.channel.typing():
        question = message.content.strip()
        abstracts, citations = search_pubmed(question)
        response_text = synthesize_with_ai(question, abstracts)

        store_key = f"{message.channel.id}_{message.author.id}"
        citation_store[store_key] = citations

        await send_long_message(message.channel, response_text)

    await bot.process_commands(message)


@bot.command(name="ask")
async def ask_command(ctx, *, question: str):
    async with ctx.channel.typing():
        abstracts, citations = search_pubmed(question)
        response_text = synthesize_with_ai(question, abstracts)

        store_key = f"{ctx.channel.id}_{ctx.author.id}"
        citation_store[store_key] = citations

        await send_long_message(ctx.channel, response_text)


@bot.command(name="cite")
async def cite(ctx):
    store_key = f"{ctx.channel.id}_{ctx.author.id}"
    citations = citation_store.get(store_key, [])

    if not citations:
        await ctx.send("No recent citations found. Ask a question first, then type !cite.")
        return

    citation_text = "**Sources from your last question:**\n\n"
    for i, citation in enumerate(citations, 1):
        citation_text += f"{i}. {citation}\n\n"

    await send_long_message(ctx.channel, citation_text)


bot.run(DISCORD_TOKEN)

