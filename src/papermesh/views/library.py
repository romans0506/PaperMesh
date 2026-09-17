"""Library page: saved topics and papers with reading status and notes."""

import streamlit as st

from papermesh import components
from papermesh.library import STATUSES
from papermesh.views.common import get_library, html, widget_id


def update_status(key: str, widget_key: str) -> None:
    get_library().set_status(key, st.session_state[widget_key])


def update_note(key: str, widget_key: str) -> None:
    get_library().set_note(key, st.session_state[widget_key])
    st.toast("Note saved", icon=":material/check:")


def remove_paper(key: str) -> None:
    get_library().remove_paper(key)
    st.toast("Removed from your library", icon=":material/bookmark_remove:")


def remove_topic(query: str) -> None:
    get_library().remove_topic(query)
    st.toast("Topic removed", icon=":material/bookmark_remove:")


def render_topics(topics: list[dict]) -> None:
    html(components.section_html("Topics", "Saved searches"))
    if not topics:
        html(components.empty_state_html(
            "No saved topics yet",
            "Search for something and choose Save topic to keep its overview here.",
        ))
        return

    with st.container(horizontal=True, gap="medium", key="pmtopics"):
        for topic in topics:
            topic_id = widget_id(topic["query"])
            with st.container(key=f"pmtopic-{topic_id}"):
                html(components.topic_card_html(topic))
                with st.container(horizontal=True, gap="small"):
                    if st.button("Open", icon=":material/arrow_outward:", key=f"open-{topic_id}", type="primary"):
                        from papermesh.views.pages import SEARCH  # late import: pages imports this module

                        st.switch_page(SEARCH, query_params={"q": topic["display"]})
                    st.button(
                        "Remove", key=f"rmtopic-{topic_id}", type="tertiary",
                        on_click=remove_topic, args=(topic["query"],),
                    )


def render_papers(counts: dict[str, int]) -> None:
    html(components.section_html("Papers", "Reading list"))
    total = sum(counts.values())
    if not total:
        html(components.empty_state_html(
            "No saved papers yet",
            "Use Save on any paper card or reading-order step to add it here.",
        ))
        return

    labels = {"all": f"All {total}"} | {s: f"{components.STATUS_LABELS[s]} {counts[s]}" for s in STATUSES}
    selected = st.segmented_control(
        "Filter", list(labels), format_func=labels.get, default="all", required=True,
        label_visibility="collapsed", key="library_filter",
    )
    papers = get_library().list_papers(status=None if selected == "all" else selected)
    if not papers:
        html(components.empty_state_html("Nothing here", "No saved papers have this status yet."))
        return

    for paper in papers:
        paper_id = widget_id(paper["key"])
        with st.container(key=f"pmsaved-{paper_id}"):
            html(components.saved_paper_html(paper))
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                status_key = f"status-{paper_id}"
                st.segmented_control(
                    "Status", list(STATUSES), format_func=components.STATUS_LABELS.get,
                    default=paper["status"], required=True, label_visibility="collapsed",
                    key=status_key, on_change=update_status, args=(paper["key"], status_key),
                )
                with st.popover("Note", icon=":material/edit_note:", type="tertiary"):
                    note_key = f"note-{paper_id}"
                    st.text_area(
                        "Note", value=paper["note"], key=note_key, label_visibility="collapsed",
                        placeholder="What's worth remembering about this paper?",
                        on_change=update_note, args=(paper["key"], note_key),
                    )
                    st.caption("Saved when you click outside the box.")
                st.button(
                    "Remove", icon=":material/delete:", key=f"rmpaper-{paper_id}", type="tertiary",
                    on_click=remove_paper, args=(paper["key"],),
                )


def render() -> None:
    library = get_library()
    topics = library.list_topics()
    counts = library.status_counts()

    total = sum(counts.values())
    subtitle = (
        f"{len(topics)} topic{'s' if len(topics) != 1 else ''} · "
        f"{total} paper{'s' if total != 1 else ''} · {counts['read']} read"
    )
    html(components.page_header_html("Library", "Your research, saved.", subtitle))
    render_topics(topics)
    render_papers(counts)
