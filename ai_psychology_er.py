from graphviz import Digraph

dot = Digraph("ER_Diagram", format="png", engine="neato")
dot.attr(
    overlap="false",
    splines="line",
    sep="+4",
    fontname="Microsoft YaHei",
    dpi="180",
)
dot.attr("edge", fontname="Microsoft YaHei", fontsize="10", arrowhead="none")
dot.attr("node", fontname="Microsoft YaHei")

ENTITY_FILL = "#F8F4D8"
REL_FILL = "#F6E2B3"


def entity(name, label, pos):
    dot.node(
        name, label,
        shape="box",
        style="filled",
        fillcolor=ENTITY_FILL,
        color="black",
        width="1.0",
        height="0.46",
        fontsize="12",
        pos=f"{pos}!",
    )


def relation(name, label, pos):
    dot.node(
        name, label,
        shape="diamond",
        style="filled",
        fillcolor=REL_FILL,
        color="black",
        width="0.86",
        height="0.58",
        fontsize="10",
        pos=f"{pos}!",
    )


def attr(name, label, pos):
    dot.node(
        name, label,
        shape="ellipse",
        style="filled",
        fillcolor="white",
        color="black",
        fontsize="9",
        width="0.95",
        height="0.34",
        margin="0.02,0.01",
        pos=f"{pos}!",
    )


def rel_edge(a, b, label):
    dot.edge(a, b, label=label, fontsize="10", len="0.7")


def attr_edge(owner, name):
    dot.edge(owner, name, len="0.45")


entity("User", "User", "0,0")
entity("Profile", "Profile", "-3.2,2.05")
entity("Chat", "Chat", "-3.8,0")
entity("Diary", "Diary", "-3.2,-2.05")
entity("Assessment", "Assessment", "3.2,2.05")
entity("Report", "Report", "5.7,2.05")
entity("Risk", "Risk", "3.8,0")
entity("Knowledge", "Knowledge", "3.2,-2.05")

relation("has", "has", "-1.7,1.1")
relation("creates", "creates", "-1.9,0")
relation("writes", "writes", "-1.7,-1.1")
relation("takes", "takes", "1.7,1.1")
relation("generates", "generates", "4.45,2.05")
relation("triggers", "triggers", "1.9,0")
relation("manages", "manages", "1.7,-1.1")

rel_edge("User", "has", "1")
rel_edge("has", "Profile", "1")
rel_edge("User", "creates", "1")
rel_edge("creates", "Chat", "n")
rel_edge("User", "writes", "1")
rel_edge("writes", "Diary", "n")
rel_edge("User", "takes", "1")
rel_edge("takes", "Assessment", "n")
rel_edge("Assessment", "generates", "1")
rel_edge("generates", "Report", "1")
rel_edge("User", "triggers", "1")
rel_edge("triggers", "Risk", "n")
rel_edge("User", "manages", "1")
rel_edge("manages", "Knowledge", "n")

attr("u_id", "userID", "0,0.82")
attr("u_name", "username", "-1.02,0.38")
attr("u_role", "role", "1.02,0.38")
attr_edge("User", "u_id")
attr_edge("User", "u_name")
attr_edge("User", "u_role")

attr("p_name", "realName", "-4.5,2.75")
attr("p_gender", "gender", "-4.75,2.05")
attr("p_age", "age", "-4.5,1.35")
attr_edge("Profile", "p_name")
attr_edge("Profile", "p_gender")
attr_edge("Profile", "p_age")

attr("c_id", "messageID", "-5.2,0.55")
attr("c_content", "content", "-5.2,-0.55")
attr_edge("Chat", "c_id")
attr_edge("Chat", "c_content")

attr("d_title", "title", "-4.5,-1.35")
attr("d_mood", "moodTag", "-4.5,-2.75")
attr_edge("Diary", "d_title")
attr_edge("Diary", "d_mood")

attr("a_type", "type", "2.4,2.8")
attr("a_score", "score", "3.2,2.9")
attr("a_level", "riskLevel", "4.05,2.8")
attr_edge("Assessment", "a_type")
attr_edge("Assessment", "a_score")
attr_edge("Assessment", "a_level")

attr("r_id", "reportID", "6.95,2.65")
attr("r_content", "content", "6.95,1.45")
attr_edge("Report", "r_id")
attr_edge("Report", "r_content")

attr("rk_level", "riskLevel", "5.2,0.55")
attr("rk_status", "status", "5.2,-0.55")
attr_edge("Risk", "rk_level")
attr_edge("Risk", "rk_status")

attr("k_title", "title", "4.5,-1.35")
attr("k_time", "uploadTime", "4.6,-2.75")
attr_edge("Knowledge", "k_title")
attr_edge("Knowledge", "k_time")

dot.render("ai_psychology_er_diagram", view=False)
print("已生成 ER 图：ai_psychology_er_diagram.png")
