# Q3 Roadmap Sync — 2026-04-22

**Attendees:** Alice Chen (VP Eng), Bob Patel (PM), Carol Johnson (Design Lead), Dan Kim (CTO)

---

**Alice:** Thanks for joining. Headline numbers from Q2: we shipped the multi-tenant rollout, NPS is up 9 points, and SLA breaches dropped 40%. Big wins. But the agent-eval feature is still 6 weeks behind plan.

**Dan:** Right. So what's the proposal for Q3?

**Bob:** Three priorities. One — finish agent-eval, target October 1. Two — start the Slack integration, scoped at 4 weeks for a v1. Three — pay down the auth-middleware tech debt; legal flagged it again last week.

**Carol:** I want to add a fourth: redesign the trace inspector. We've gotten the same complaint from three enterprise customers in two weeks. The current view doesn't scale past 20 spans.

**Dan:** Agreed on prioritization, but we can't do all four. I'll be the unpopular voice — Slack integration slips to Q4. Auth-middleware first because compliance won't wait, then agent-eval, then trace inspector. Slack pushed.

**Alice:** OK, decision recorded: Slack moves to Q4. Bob, can you own the auth-middleware migration plan? Need a draft RFC by Monday April 28.

**Bob:** I'll have it Monday.

**Carol:** I'll start the trace inspector mockups this week — happy to circulate v1 by Friday April 26.

**Alice:** And I'll own agent-eval delivery against October 1. Dan, I need budget approval for two more SDETs to make that date.

**Dan:** Approved up to $250k for contractor SDETs. Get the requisitions in this week.

**Bob:** One last thing — the customer advisory board meeting is May 12. Carol, can you prep the trace inspector demo by then?

**Carol:** Tight but yes. I'll plan to demo by May 11 to leave buffer.

**Alice:** Great. Anything else?

**Dan:** Nothing from me.

**Alice:** Adjourned. Thanks all.
