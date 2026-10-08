# LinkedIn document (PDF carousel) — plan, 2026-10-08

**Written at Jeff's instruction** ("Create Plan to add LinkedIn document (PDF carousel)"). Nothing
here is built. The type exists in both repositories as one of the thirteen disabled kinds
(`content-creator-v2/src/lib/content-types.ts:60-79`; GeekBackend
`GccGenerationCoordinator.cs:174-193` refuses it as "disabled pending a written, approved resolve
plan for its content-type quality"). This is that plan; it is enabled when Jeff approves it and the
stages below are built.

## What it is, and what it is not

A LinkedIn document post is a PDF that LinkedIn renders as a swipeable carousel: one page per
slide, portrait or square, read on a phone. For this app it is **a first-class generated type,
written from the brief and the retrieved evidence like the blog is** — not a repurposing of a page.
Jeff, 2026-10-08: nothing the app has produced is good enough to repurpose yet (short-form pieces
such as the cold email may be the exception); the document stands on its own.

It is grounded like every long-form type: the partner passages and the publisher's positions are
its evidence, figures come only from that evidence, and it carries no quotation (the pillar's and
blog's rule, not the tool page's). It ends on the fixed consultation line, as every page does.

## The shape

- 8–12 slides. Slide 1 is the cover: the keyword's promise in one line and a sub-line. Slides 2–n
  each carry one point: a heading of at most eight words and two to four lines of at most twenty
  words each. The last slide is the closing: "Answer these questions when booking your free
  consultation" with the diagnosis questions, and the scheduler URL as **text** — links inside a
  LinkedIn document are not reliably clickable, so the URL is printed.
- One `ContentDocument`: the lede is the cover (heading + its lines), each slide is a `Section`
  with `Tag = "h2"` and `TextParagraph`s only — no lists, no quotes, no children, no hrefs except
  the closing's scheduler href, which the renderer prints as text. `SectionHtmlRenderer` stays the
  only HTML producer; the HTML view is the slide preview.
- Portrait 1080 × 1350 (4:5), which LinkedIn shows largest on a phone; one slide per PDF page.

## Stages

**L1 — The type (GeekBackend).** A `linkedin-document` entry in the content-type registry with its
own prompt: the brief's framing (core problem, the rows' problems, the automation to pitch), the
evidence block cut to the partner passages and positions (the research block as the blog gets it,
not PARTNER DATA), the slide shape above stated once, the brand voice, the length rule stated as
the slide and line limits — not words — and no CTA language (the page builds the closing). One
model call returns the `sections` array (slides); the cover is a second, small call or the first
section of the same call — decided in L1 by what the contract test shows is more reliable.
Guards: `numbers`, `currency`, `questions-quiz`, `no-quotation`, `partner-mentions` as a gap, a new
`slide-shape` check (count, heading length, line length) that refuses. No link guard beyond the
closing. Tests: the prompt names the shape and nothing else; a scripted reply over the limits is
refused by name.

**L2 — The PDF (GeekBackend).** A renderer that takes the `ContentDocument` and returns PDF bytes:
one page per slide, 1080 × 1350 pt-equivalent, the publisher's colours and logo from
`CompanyProfileOptions`, the cover and closing laid out differently from the body slides. Two
candidates, Jeff's decision 1: **QuestPDF** in GeekAPI (a NuGet dependency; layout in code, fonts
embedded, deterministic output, the PDF stored as a version artifact and served by
`GET …/versions/{id}/export/pdf`), or the browser printing the HTML slide preview to PDF (no
backend change, but fonts, page breaks and colours are the browser's, and nothing is stored).
Recommendation: QuestPDF — the deliverable should exist on the server, not depend on whoever
presses Print. Tests: a fixed document renders to a PDF with n pages; the closing page carries the
URL as text.

**L3 — Enable it (both repositories, paired).** Remove `linkedin-document` from the disabled list
in `content-types.ts` and from the backend refusal; the type allow-list of the audit plan's X8
applies. The output tab shows the slides as the HTML preview with a "Download PDF" control calling
L2's route. The brief needs nothing new: the document reads the framing and the evidence the
blog reads.

**L4 — Proof.** One Generate on `cc480d8c` with the document requested alongside the blog: the
slides read by Jeff; every figure on a slide found in the evidence; the PDF uploads to a LinkedIn
document post (PDF, under 100 MB, under 300 pages) and renders as a carousel on a phone.

## Order and cost

L1 (half a day) → L2 (half a day with QuestPDF; an hour with browser print) → L3 (an hour) → L4
(Jeff). L1 does not wait on anything; L2 can be built in parallel once the document shape is fixed.

## Decisions for Jeff

| # | Decision | Recommendation |
|---|---|---|
| 1 | PDF renderer: QuestPDF in GeekAPI, or the browser's print | QuestPDF — the PDF is the deliverable and should be stored with the version |
| 2 | Slide count and aspect | 8–12 slides, 1080 × 1350 portrait |
| 3 | The closing slide: the diagnosis questions and the scheduler URL as text | Yes — the same closing as every page, printed not linked |
| 4 | Quotations on a slide | None, the pillar/blog rule; a verified partner quotation as its own slide can be added after `/v1/verify` is wired (audit X2) |

## Rules that bind this plan

The model returns content, never markup (CLAUDE.md §1b); the renderer is the only thing that
makes pages. Fail closed: a reply over the slide limits is refused, not trimmed; a figure not in the
evidence refuses the document. Nothing in it is quoted from the brief. Contract changes land in
paired commits.
