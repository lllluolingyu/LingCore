---
description: Put an event on my calendar from a plain-language description.
argument_hint: "[e.g. dentist next Tuesday 3pm for 45 minutes, remind me an hour before]"
---
Add to my calendar: $ARGUMENTS

Resolve relative dates and times ("next Tuesday", "tonight") against the
`[clock]` line. If the date, the start time, or whether it is all day is
genuinely ambiguous, ask one short question instead of guessing. Use the
default length when no duration is given.

Then save it with `calendar_add`, including the location, reminders and repeat
rule if I gave them. Reply with the event as saved (day, date, time, timezone)
and mention any overlap with an existing event that the result reports.
