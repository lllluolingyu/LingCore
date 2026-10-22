---
description: Show today's agenda, upcoming appointments, and relevant priorities.
argument_hint: "[optional focus, e.g. work or preparing for my trip]"
---
Show me today's agenda. Optional focus: $ARGUMENTS

Use the latest `[clock]` line for the local date, time, and timezone. Call
`calendar_list` with both `start` and `end` set to that date; its default
window is much longer than one day. Use saved memory for relevant preferences
and commitments. Search local notes only when needed to resolve a reference.

Give a short agenda headed by the weekday, date, and timezone: all-day events,
then timed appointments in order. If the calendar reports a different timezone,
state it explicitly rather than relabeling its times. Distinguish appointments
already passed from those still upcoming. Include known deadlines or
preparation items, and suggest at most three priorities grounded in my saved
information. Keep suggestions separate from scheduled events; do not invent
tasks or assume an appointment was completed just because its time passed.

Mention overlapping events and any incomplete-coverage warning. An empty or
partial local calendar does not establish availability in an external calendar.
If nothing is saved, say so without making up a schedule. Do not create,
update, or remove events, write memory, search the web, or promise delivered
reminders merely to produce this agenda.
