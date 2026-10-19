---
description: Plan a day, a week or a task into concrete, timed steps.
argument_hint: "[what to plan, e.g. tomorrow, this weekend, moving house]"
---
Help me plan: $ARGUMENTS

Use the `[clock]` line for today's date. First check what is already on my
calendar for that period with `calendar_list`, and read memory for preferences
or commitments that matter. If the goal, time window or constraints are
unclear, ask me at most three short questions before planning.

Then give a plan as a short ordered list: what to do, when, roughly how long,
and anything to prepare or book beforehand. Keep realistic gaps for travel and
breaks, and flag conflicts with saved events. Look up opening hours, travel
times or prices with `search_web`/`fetch_url` only when the plan depends on
them, and cite the sources.

End by offering to save the timed items with `calendar_add`; do not save them
until I agree.
