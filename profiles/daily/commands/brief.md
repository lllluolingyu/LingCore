---
description: A short, sourced briefing on a topic or question.
argument_hint: "[topic or question]"
---
Give me a briefing on: $ARGUMENTS

Research it with `search_web`, then verify the key facts on reliable,
preferably primary or official pages with `fetch_url`. Use the `[clock]` line
to judge how current each source is and say when information may be outdated.

Format:
1. **Bottom line**: two or three sentences answering what matters most.
2. **Key points**: up to six bullets, each with its source URL.
3. **Open questions**: what is disputed, uncertain or still changing.

Keep it under about 300 words unless I ask for more. Separate confirmed facts
from estimates and opinions, and never present remembered facts as live
results.
