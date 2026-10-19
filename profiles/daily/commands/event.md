---
description: Turn a plain-language description into a calendar event file.
argument_hint: "[e.g. dentist next Tuesday 3pm for 45 minutes, remind me an hour before]"
---
Add to my calendar: $ARGUMENTS

Resolve relative dates and times ("next Tuesday", "tonight") against the
`[clock]` line. If the date, the start time, or whether it is all day is
genuinely ambiguous, ask one short question instead of guessing. Use the
default length when no duration is given. Check `calendar_list` for that day and
mention any overlap.

Then save it with `calendar_add`, including the location, reminders and repeat
rule if I gave them. Reply with the event as saved (day, date, time, timezone)
and the file path, and remind me that opening or importing the file adds it to
my calendar app.
