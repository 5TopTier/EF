
import os
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import datetime as dt
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks
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
MEMORY_FILE = "memory.json"
LAST_RESEARCH_FILE = "last_research_update.json"
RESEARCH_UPDATE_TIME = dt.time(hour=10, minute=0, tzinfo=ZoneInfo("America/Chicago"))


def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


user_memory = load_json_file(MEMORY_FILE, {})

SYSTEM_PROMPT = """
IDENTITY
You are the Euphoria Fit Research Bot.

Your purpose is to answer health and fitness questions using the best
available evidence from research literature and high-quality web sources.

You may be provided with:
- peer-reviewed research findings
- trusted web-based evidence
- practical context from the user's stored profile

Base your response on the retrieved evidence provided to you.
Prioritize peer-reviewed research over general web information.
Do not make claims that go beyond what the retrieved evidence supports.

RESPONSE RULES
- Prioritize meta-analyses and systematic reviews above all else
- Prefer randomized trials over observational studies
- Use broader web evidence when research is sparse, indirect, or not directly practical
- If evidence is weak, say so directly
- Never present a single source as definitive
- If findings conflict, explain why they may conflict
- Do not diagnose injuries, conditions, or symptoms
- Do not recommend medications or therapeutic doses
- For medical questions, recommend consulting a physician or physical therapist
- Plain language, intellectually honest, no hype

STYLE RULES
You may be given a RESPONSE_STYLE:
- concise
- deep_dive
- myth_check
- coaching
- conflict_analysis

FORMAT
1. WHAT THE EVIDENCE SHOWS
2. EVIDENCE QUALITY
3. CONFLICTING FINDINGS (only if needed)
4. PRACTICAL TAKEAWAY
5. Want the full citations? Type !cite
""".strip()


def get_user_profile(user_id: str):
    if user_id not in user_memory:
        user_memory[user_id] = {
            "goals": [],
            "experience_level": "unknown",
            "recent_topics": [],
            "recent_styles": [],
            "last_questions": [],
            "training_profile": {
                "goal": "general_fitness",
                "days_per_week": 3,
                "session_length_min": 60,
                "equipment": "full_gym",
                "limitations": "none reported",
                "style_preference": "general",
            },
            "updated_at": None,
        }
    return user_memory[user_id]


def save_memory():
    save_json_file(MEMORY_FILE, user_memory)


def save_profile(user_id: str):
    profile = get_user_profile(user_id)
    profile["updated_at"] = datetime.utcnow().isoformat()
    save_memory()


def update_user_memory(user_id: str, question: str, topic: str, style: str):
    profile = get_user_profile(user_id)
    q = question.lower()

    if any(word in q for word in ["beginner", "new to lifting", "just started", "newbie"]):
        profile["experience_level"] = "beginner"
    elif any(word in q for word in ["advanced", "peaking", "periodization", "elite", "intermediate"]):
        profile["experience_level"] = "advanced"

    goal_map = {
        "fat_loss": ["fat loss", "lose weight", "cut", "calorie deficit", "diet"],
        "muscle_gain": ["hypertrophy", "build muscle", "muscle gain", "bulk"],
        "strength": ["strength", "powerlifting", "1rm", "squat", "bench", "deadlift"],
        "recovery": ["recovery", "sleep", "fatigue", "soreness", "deload"],
        "health": ["health", "blood pressure", "cholesterol", "longevity"],
    }

    for goal, keywords in goal_map.items():
        if any(k in q for k in keywords) and goal not in profile["goals"]:
            profile["goals"].append(goal)

    if topic:
        profile["recent_topics"].append(topic)
        profile["recent_topics"] = profile["recent_topics"][-8:]

    if style:
        profile["recent_styles"].append(style)
        profile["recent_styles"] = profile["recent_styles"][-5:]

    profile["last_questions"].append(question[:200])
    profile["last_questions"] = profile["last_questions"][-5:]

    save_profile(user_id)


def classify_topic(question: str) -> str:
    q = question.lower()

    rules = {
        "hypertrophy": ["hypertrophy", "muscle growth", "build muscle", "volume", "sets per week", "failure training"],
        "strength": ["strength", "1rm", "powerlifting", "squat", "bench", "deadlift"],
        "fat_loss": ["fat loss", "lose fat", "lose weight", "cut", "deficit", "appetite"],
        "nutrition": ["protein", "carbs", "fats", "meal timing", "calories", "diet"],
        "supplements": ["creatine", "caffeine", "beta alanine", "supplement", "pre workout"],
        "recovery": ["recovery", "sleep", "soreness", "fatigue", "deload"],
        "cardio": ["cardio", "running", "vo2", "zone 2", "aerobic", "hiit"],
        "injury_pain": ["pain", "injury", "hurt", "strain", "sprain", "tendon", "physical therapy"],
        "body_composition": ["body fat", "lean mass", "recomp", "body composition"],
        "training_plan": ["plan", "program", "split", "routine", "workout plan"],
    }

    for topic, keywords in rules.items():
        if any(k in q for k in keywords):
            return topic

    return "general_fitness"


def choose_response_style(question: str, profile: dict) -> str:
    q = question.lower()

    if any(x in q for x in ["myth", "is it true", "does x really", "debunk", "fake", "bro science"]):
        return "myth_check"

    if any(x in q for x in ["conflict", "mixed evidence", "studies disagree", "vs", "better than"]):
        return "conflict_analysis"

    if any(x in q for x in ["how should i", "what should i do", "practically", "best way to apply"]):
        return "coaching"

    if any(x in q for x in ["explain deeply", "deep dive", "detailed", "thorough"]):
        return "deep_dive"

    recent = profile.get("recent_styles", [])
    if recent and recent[-1] == "concise":
        return "coaching"

    return "concise"


def build_pubmed_query(question: str, topic: str) -> str:
    topic_boosts = {
        "hypertrophy": "(muscle hypertrophy OR resistance training OR training volume)",
        "strength": "(maximal strength OR resistance training OR one repetition maximum)",
        "fat_loss": "(fat loss OR body weight OR calorie deficit OR appetite)",
        "nutrition": "(dietary protein OR energy intake OR meal timing OR sports nutrition)",
        "supplements": "(creatine OR caffeine OR ergogenic aids OR dietary supplements)",
        "recovery": "(sleep OR fatigue OR recovery OR muscle soreness)",
        "cardio": "(aerobic exercise OR HIIT OR endurance OR VO2max)",
        "injury_pain": "(musculoskeletal pain OR injury OR rehabilitation OR physical therapy)",
        "body_composition": "(body composition OR lean mass OR fat mass)",
        "training_plan": "(resistance training OR exercise program OR exercise prescription)",
        "general_fitness": "(exercise OR training OR fitness)",
    }

    boost = topic_boosts.get(topic, "(exercise OR training OR fitness)")
    return f"({question}) AND {boost}"


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
        "reldate": 3650,
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

        evidence_items = []
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
                evidence_items.append(
                    f"RESEARCH ITEM\n"
                    f"TITLE: {title}\n"
                    f"SOURCE: {journal} ({year})\n"
                    f"DETAILS: {abstract[:1400]}"
                )
                citations.append(f'{author_str}. "{title}" {journal}. {year}. {pubmed_link}')

        return evidence_items, citations

    except Exception as e:
        print(f"search_pubmed error: {e}")
        return [], []


def should_use_web_fallback(question: str, research_items: list) -> bool:
    q = question.lower()

    if not research_items:
        return True

    if len(research_items) < 2:
        return True

    practical_terms = [
        "best", "worth it", "how should i", "what should i do", "practical",
        "sample plan", "routine", "split", "exercise selection", "equipment",
        "home gym", "what exercises", "weekly plan"
    ]

    return any(term in q for term in practical_terms)


def build_context_block(profile: dict, topic: str, style: str) -> str:
    goals = ", ".join(profile.get("goals", [])) if profile.get("goals") else "none recorded"
    recent_topics = ", ".join(profile.get("recent_topics", [])[-3:]) if profile.get("recent_topics") else "none"
    experience = profile.get("experience_level", "unknown")
    training = profile.get("training_profile", {})

    return f"""
USER CONTEXT
- Experience level: {experience}
- Recurring goals: {goals}
- Recent topics: {recent_topics}
- Current classified topic: {topic}
- RESPONSE_STYLE: {style}

TRAINING PROFILE
- Goal: {training.get("goal", "general_fitness")}
- Days/week: {training.get("days_per_week", 3)}
- Session length: {training.get("session_length_min", 60)} minutes
- Equipment: {training.get("equipment", "full_gym")}
- Limitations: {training.get("limitations", "none reported")}
- Style preference: {training.get("style_preference", "general")}

Use this only to make the answer more relevant and less repetitive.
Do not invent facts about the user.
""".strip()


def synthesize_with_ai(question, research_items, profile, topic, style, allow_web=False):
    research_context = "\n\n".join(research_items) if research_items else "No strong research items were retrieved."

    context_block = build_context_block(profile, topic, style)

    user_message = f"""
{context_block}

USER QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{research_context}

Additional instructions:
- Prioritize research evidence over general web evidence.
- If direct research is missing or incomplete, use web search to fill practical or informational gaps.
- Do not say you are using "abstracts".
- Refer to the evidence as research, evidence, literature, or available evidence.
- If the evidence is limited, say so clearly.
- If this is an injury or symptom question, do not diagnose.
- Keep the wording natural and not repetitive.

Answer now.
""".strip()

    request_kwargs = {
        "model": "gpt-5.2",
        "instructions": SYSTEM_PROMPT,
        "input": user_message,
    }

    if allow_web:
        request_kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**request_kwargs)
    return (response.output_text or "").strip()


def infer_goal_from_text(text: str) -> str:
    q = text.lower()
    if any(x in q for x in ["fat loss", "lose fat", "lose weight", "cut", "deficit"]):
        return "fat_loss"
    if any(x in q for x in ["hypertrophy", "muscle gain", "build muscle", "bulk"]):
        return "muscle_gain"
    if any(x in q for x in ["strength", "powerlifting", "1rm", "squat", "bench", "deadlift"]):
        return "strength"
    if any(x in q for x in ["recomp", "body recomposition"]):
        return "recomp"
    return "general_fitness"


def normalize_equipment(text: str) -> str:
    q = text.lower().strip()
    if "full gym" in q or q == "gym":
        return "full_gym"
    if "dumbbell" in q or "db" in q:
        return "dumbbells"
    if "home" in q:
        return "home"
    if "machine" in q:
        return "machines"
    if "bodyweight" in q:
        return "bodyweight"
    return q or "unknown"


def get_training_profile(user_id: str):
    profile = get_user_profile(user_id)
    if "training_profile" not in profile:
        profile["training_profile"] = {
            "goal": "general_fitness",
            "days_per_week": 3,
            "session_length_min": 60,
            "equipment": "full_gym",
            "limitations": "none reported",
            "style_preference": "general",
        }
    return profile["training_profile"]


def save_training_profile(user_id: str, updates: dict):
    profile = get_user_profile(user_id)
    training = get_training_profile(user_id)
    training.update(updates)
    save_profile(user_id)


def build_plan_framework(goal: str, days_per_week: int, equipment: str, style_preference: str) -> dict:
    goal = goal or "general_fitness"
    equipment = equipment or "full_gym"
    style_preference = style_preference or "general"

    if days_per_week <= 2:
        split = ["Full Body A", "Full Body B"]
    elif days_per_week == 3:
        split = ["Full Body A", "Full Body B", "Full Body C"]
    elif days_per_week == 4:
        if goal == "strength":
            split = ["Upper Strength", "Lower Strength", "Upper Volume", "Lower Volume"]
        else:
            split = ["Upper 1", "Lower 1", "Upper 2", "Lower 2"]
    elif days_per_week == 5:
        if goal == "strength":
            split = ["Squat Focus", "Bench Focus", "Pull Focus", "Upper Volume", "Lower Volume"]
        else:
            split = ["Upper", "Lower", "Push", "Pull", "Legs"]
    else:
        split = ["Push", "Pull", "Legs", "Upper", "Lower", "Optional Weak Point / Conditioning"]

    return {
        "goal": goal,
        "days_per_week": days_per_week,
        "equipment": equipment,
        "style_preference": style_preference,
        "split": split,
    }


def build_plan_context(user_id: str) -> str:
    profile = get_user_profile(user_id)
    training = get_training_profile(user_id)

    goals = ", ".join(profile.get("goals", [])) if profile.get("goals") else "none recorded"
    recent_topics = ", ".join(profile.get("recent_topics", [])[-4:]) if profile.get("recent_topics") else "none"

    return f"""
USER PROFILE
- Experience level: {profile.get("experience_level", "unknown")}
- Recurring goals history: {goals}
- Recent topics: {recent_topics}

TRAINING PROFILE
- Primary goal: {training.get("goal", "general_fitness")}
- Days per week: {training.get("days_per_week", 3)}
- Session length: {training.get("session_length_min", 60)} minutes
- Equipment: {training.get("equipment", "full_gym")}
- Limitations: {training.get("limitations", "none reported")}
- Style preference: {training.get("style_preference", "general")}
""".strip()


def generate_training_plan_with_ai(user_id: str, request_text: str) -> str:
    training = get_training_profile(user_id)
    framework = build_plan_framework(
        training.get("goal", "general_fitness"),
        int(training.get("days_per_week", 3)),
        training.get("equipment", "full_gym"),
        training.get("style_preference", "general"),
    )
    context_block = build_plan_context(user_id)

    prompt = f"""
{context_block}

PLAN FRAMEWORK
- Goal: {framework["goal"]}
- Days per week: {framework["days_per_week"]}
- Equipment: {framework["equipment"]}
- Split suggestion: {", ".join(framework["split"])}

USER REQUEST
{request_text}

Generate an evidence-informed training suggestion and sample plan.
Use research-first reasoning, but use web search if needed for practical exercise selection or implementation details.
Rules:
- Do not diagnose injuries or prescribe rehab.
- Keep it practical and personalized to the profile.
- Make it a starting point, not a claim of perfection.
- Include progression, exercise substitutions, and when to adjust.
- Support fat loss, muscle gain, strength, recomp, and general fitness.
- If the user's limitations sound medical, say they should consult a physician or physical therapist.

Output format:
1. WHO THIS PLAN FITS
2. WEEKLY SPLIT
3. WORKOUTS BY DAY
4. PROGRESSION RULE
5. CARDIO / RECOVERY GUIDANCE
6. WHEN TO ADJUST
""".strip()

    response = client.responses.create(
        model="gpt-5.2",
        instructions=(
            "You are the Euphoria Fit Research Bot in coaching mode. "
            "Give practical, personalized, evidence-informed sample training plans. "
            "Use plain language. No hype. No medical diagnosis."
        ),
        tools=[{"type": "web_search"}],
        input=prompt,
    )

    return (response.output_text or "").strip()


def search_latest_pubmed(query, max_results=8):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"

    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "pub date",
        "datetype": "pdat",
        "reldate": 7,
    }

    try:
        search_resp = requests.get(search_url, params=search_params, timeout=20)
        search_resp.raise_for_status()
        pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

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

        papers = []

        for article in root.findall(".//PubmedArticle"):
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

            abstract_texts = article.findall(".//AbstractText")
            abstract_parts = []
            for t in abstract_texts:
                text = "".join(t.itertext()).strip()
                if text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
            pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            if title and pmid:
                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "authors": author_str,
                    "summary": abstract[:500] if abstract else "No summary available.",
                    "link": pubmed_link,
                })

        return papers

    except Exception as e:
        print(f"search_latest_pubmed error: {e}")
        return []


def format_research_update_message(papers):
    if not papers:
        return (
            "**Daily Research Update**\n\n"
            "No new peer-reviewed papers were found in the latest search window."
        )

    lines = ["**Daily Research Update**\n"]

    for i, paper in enumerate(papers, 1):
        lines.append(
            f"**{i}. {paper['title']}**\n"
            f"{paper['authors']} — *{paper['journal']}* ({paper['year']})\n"
            f"{paper['link']}\n"
            f"**Why it matters:** {paper['summary']}\n"
        )

    return "\n".join(lines)


def get_new_research_papers():
    query = (
        "(resistance training OR hypertrophy OR strength OR fat loss OR sports nutrition) "
        "AND (systematic[sb] OR meta-analysis[pt] OR randomized controlled trial[pt])"
    )

    papers = search_latest_pubmed(query, max_results=8)
    last_data = load_json_file(LAST_RESEARCH_FILE, {"pmids": []})
    seen_pmids = set(last_data.get("pmids", []))

    new_papers = [paper for paper in papers if paper["pmid"] not in seen_pmids]

    if papers:
        save_json_file(LAST_RESEARCH_FILE, {"pmids": [paper["pmid"] for paper in papers]})

    return new_papers[:5]


async def send_long_message(channel, text, limit=1900):
    if not text:
        await channel.send("I could not generate a response.")
        return

    chunks = [text[i:i + limit] for i in range(0, len(text), limit)]
    for chunk in chunks:
        await channel.send(chunk)


async def handle_question(message_channel, user_id: str, question: str, store_key: str):
    profile = get_user_profile(user_id)
    topic = classify_topic(question)
    style = choose_response_style(question, profile)
    pubmed_query = build_pubmed_query(question, topic)

    research_items, citations = search_pubmed(pubmed_query)
    use_web = should_use_web_fallback(question, research_items)

    response_text = synthesize_with_ai(
        question=question,
        research_items=research_items,
        profile=profile,
        topic=topic,
        style=style,
        allow_web=use_web,
    )

    citation_store[store_key] = citations
    update_user_memory(user_id, question, topic, style)

    await send_long_message(message_channel, response_text)


@tasks.loop(time=RESEARCH_UPDATE_TIME)
async def daily_research_updates():
    channel = discord.utils.get(bot.get_all_channels(), name="research-updates")
    if channel is None:
        print("Could not find #research-updates channel")
        return

    papers = get_new_research_papers()
    message = format_research_update_message(papers)
    await send_long_message(channel, message)


@daily_research_updates.before_loop
async def before_daily_research_updates():
    await bot.wait_until_ready()


@bot.event
async def on_ready():
    print(f"Euphoria Fit Research Bot is online as {bot.user}")
    if not daily_research_updates.is_running():
        daily_research_updates.start()


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

    if len(message.content.strip()) < 8:
        await bot.process_commands(message)
        return

    async with message.channel.typing():
        store_key = f"{message.channel.id}_{message.author.id}"
        await handle_question(
            message.channel,
            str(message.author.id),
            message.content.strip(),
            store_key,
        )

    await bot.process_commands(message)


@bot.command(name="ask")
async def ask_command(ctx, *, question: str):
    async with ctx.channel.typing():
        store_key = f"{ctx.channel.id}_{ctx.author.id}"
        await handle_question(
            ctx.channel,
            str(ctx.author.id),
            question.strip(),
            store_key,
        )


@bot.command(name="cite")
async def cite(ctx):
    store_key = f"{ctx.channel.id}_{ctx.author.id}"
    citations = citation_store.get(store_key, [])

    if not citations:
        await ctx.send("No stored research citations found for your last question.")
        return

    citation_text = "**Research citations from your last question:**\n\n"
    for i, citation in enumerate(citations, 1):
        citation_text += f"{i}. {citation}\n\n"

    citation_text += (
        "Note: some answers may also use broader web evidence for practical context "
        "when direct research is limited."
    )

    await send_long_message(ctx.channel, citation_text)


@bot.command(name="setgoal")
async def set_goal(ctx, *, goal: str):
    parsed = infer_goal_from_text(goal)
    save_training_profile(str(ctx.author.id), {"goal": parsed})
    await ctx.send(f"Saved your primary goal as **{parsed}**.")


@bot.command(name="setdays")
async def set_days(ctx, days: int):
    if days < 2 or days > 6:
        await ctx.send("Pick a number from 2 to 6.")
        return
    save_training_profile(str(ctx.author.id), {"days_per_week": days})
    await ctx.send(f"Saved your training frequency as **{days} days/week**.")


@bot.command(name="setsession")
async def set_session(ctx, minutes: int):
    if minutes < 20 or minutes > 180:
        await ctx.send("Pick a session length from 20 to 180 minutes.")
        return
    save_training_profile(str(ctx.author.id), {"session_length_min": minutes})
    await ctx.send(f"Saved your session length as **{minutes} minutes**.")


@bot.command(name="setequipment")
async def set_equipment(ctx, *, equipment: str):
    parsed = normalize_equipment(equipment)
    save_training_profile(str(ctx.author.id), {"equipment": parsed})
    await ctx.send(f"Saved your equipment as **{parsed}**.")


@bot.command(name="setlimits")
async def set_limits(ctx, *, limitations: str):
    save_training_profile(str(ctx.author.id), {"limitations": limitations[:300]})
    await ctx.send("Saved your limitations note.")


@bot.command(name="setstyle")
async def set_style(ctx, *, style: str):
    style = style.lower().strip()
    allowed = {"general", "bodybuilding", "powerlifting", "athletic", "health"}
    if style not in allowed:
        await ctx.send("Use one of: general, bodybuilding, powerlifting, athletic, health")
        return
    save_training_profile(str(ctx.author.id), {"style_preference": style})
    await ctx.send(f"Saved your style preference as **{style}**.")


@bot.command(name="plan")
async def plan_command(ctx, *, request_text: str = ""):
    async with ctx.channel.typing():
        user_id = str(ctx.author.id)
        training = get_training_profile(user_id)

        if not request_text.strip():
            request_text = (
                f"Create a sample plan for a user whose primary goal is {training.get('goal', 'general_fitness')}, "
                f"training {training.get('days_per_week', 3)} days per week, "
                f"with {training.get('session_length_min', 60)} minute sessions, "
                f"using {training.get('equipment', 'full_gym')} equipment."
            )

        plan_text = generate_training_plan_with_ai(user_id, request_text)
        update_user_memory(
            user_id,
            f"PLAN REQUEST: {request_text}",
            "training_plan",
            "coaching"
        )
        await send_long_message(ctx.channel, plan_text)


@bot.command(name="sampleplan")
async def sample_plan(ctx):
    async with ctx.channel.typing():
        user_id = str(ctx.author.id)
        training = get_training_profile(user_id)
        request_text = (
            f"Create a sample weekly plan for {training.get('goal', 'general_fitness')} "
            f"with {training.get('days_per_week', 3)} days per week, "
            f"{training.get('session_length_min', 60)} minute sessions, "
            f"and {training.get('equipment', 'full_gym')} equipment."
        )
        plan_text = generate_training_plan_with_ai(user_id, request_text)
        await send_long_message(ctx.channel, plan_text)


@bot.command(name="trainingprofile")
async def training_profile_command(ctx):
    training = get_training_profile(str(ctx.author.id))
    msg = (
        f"**Your training profile**\n"
        f"- Goal: {training.get('goal', 'general_fitness')}\n"
        f"- Days/week: {training.get('days_per_week', 3)}\n"
        f"- Session length: {training.get('session_length_min', 60)} min\n"
        f"- Equipment: {training.get('equipment', 'full_gym')}\n"
        f"- Limitations: {training.get('limitations', 'none reported')}\n"
        f"- Style preference: {training.get('style_preference', 'general')}"
    )
    await ctx.send(msg)


@bot.command(name="profile")
async def profile_command(ctx):
    profile = get_user_profile(str(ctx.author.id))
    training = get_training_profile(str(ctx.author.id))

    goals = ", ".join(profile.get("goals", [])) if profile.get("goals") else "none"
    topics = ", ".join(profile.get("recent_topics", [])[-5:]) if profile.get("recent_topics") else "none"

    msg = (
        f"**Your current bot profile**\n"
        f"- Experience level: {profile.get('experience_level', 'unknown')}\n"
        f"- Goals history: {goals}\n"
        f"- Recent topics: {topics}\n"
        f"- Training goal: {training.get('goal', 'general_fitness')}\n"
        f"- Days/week: {training.get('days_per_week', 3)}\n"
        f"- Session length: {training.get('session_length_min', 60)} min\n"
        f"- Equipment: {training.get('equipment', 'full_gym')}\n"
        f"- Limitations: {training.get('limitations', 'none reported')}\n"
        f"- Style preference: {training.get('style_preference', 'general')}"
    )
    await ctx.send(msg)


@bot.command(name="resetprofile")
async def reset_profile(ctx):
    user_id = str(ctx.author.id)
    user_memory[user_id] = {
        "goals": [],
        "experience_level": "unknown",
        "recent_topics": [],
        "recent_styles": [],
        "last_questions": [],
        "training_profile": {
            "goal": "general_fitness",
            "days_per_week": 3,
            "session_length_min": 60,
            "equipment": "full_gym",
            "limitations": "none reported",
            "style_preference": "general",
        },
        "updated_at": None,
    }
    save_memory()
    await ctx.send("Your stored bot profile was reset.")


@bot.command(name="resettraining")
async def reset_training(ctx):
    save_training_profile(
        str(ctx.author.id),
        {
            "goal": "general_fitness",
            "days_per_week": 3,
            "session_length_min": 60,
            "equipment": "full_gym",
            "limitations": "none reported",
            "style_preference": "general",
        },
    )
    await ctx.send("Your training profile was reset.")


bot.run(DISCORD_TOKEN)
