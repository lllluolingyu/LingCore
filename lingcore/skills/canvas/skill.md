---
name: canvas
description: "Access Canvas LMS — list courses, assignments & due dates, announcements, and sync course files into the workspace."
requested_tools:
  - canvas_courses
  - canvas_assignments
  - canvas_announcements
  - canvas_sync
provides:
  - canvas_courses
  - canvas_assignments
  - canvas_announcements
  - canvas_sync
module: canvas_tools.py
---

You can reach the student's Canvas LMS through four tools. Canvas is the source
of truth for courses, assignments, announcements, and files — never invent due
dates, point values, or course names; read them from Canvas.

- `canvas_courses` — the student's active courses. Start here when you don't yet
  know which courses exist or need a course id for the other tools.
- `canvas_assignments` — assignments with due dates and points, sorted by due
  date. Use this for "what's due", "what's coming up", or per-course workload
  questions. Pass `course_id` to focus one class.
- `canvas_announcements` — recent announcements from instructors. Use for "any
  news from my professors" or to catch schedule/exam changes.
- `canvas_sync` — download course files (PDF/PPT/DOCX) into the workspace,
  mirroring the Canvas folder layout. It skips files already downloaded at the
  same size, so it is cheap to re-run. Sync **before** answering questions that
  depend on lecture slides or readings. Use `read_media` for PDFs/images that the
  model needs to see, and `read_file`/`search` for UTF-8 text files. Cite the
  path you used.

Guidance:
- Prefer one targeted call over many: `canvas_assignments` already spans all
  active courses, so don't loop course-by-course unless the student asked about
  one class.
- When you state a due date, give it in the student's words ("due Friday") and
  include the raw date so there's no ambiguity.
- If Canvas isn't configured (missing URL or token), say so plainly and tell the
  student which environment variable to set — don't guess at their coursework.
