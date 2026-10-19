"""S626 (5): computation-order timelines ("which token holds what, at which depth, and when is the image retrieved").
Rows measured here come from the exact interchange maps (s626_*_patch.json): the share of a source's full-swap effect that
one token (or token pair) holds at each grid layer -- drawn as a strip whose shade is the share. Rows from earlier experiments
(DAS, component patching, attention tracking) are drawn as outlined spans with their FINDINGS section.
Usage: python3 exp/s626_timeline.py OUT.json
"""
import json, sys
import numpy as np

R = "/home/storage2/ai26269345/vlm/results/r3"


def load(tag):
    return json.load(open(f"{R}/s626_{tag}_patch.json"))


def col(d, m, pick):
    u = d["units"]; X = np.array(d["maps"][m]) / d["full"][m]
    idx = [i for i, x in enumerate(u) if pick(i, x)]
    return X[:, idx].sum(1)


def answer(d, m):
    return np.array(d["maps"][m])[:, -1] / d["full"][m]


def relay(d):
    u = d["units"]; q = next(i for i, x in enumerate(u) if x.strip() == "?")
    nl = next(i for i in range(q + 1, len(u)) if u[i] == "\n")             # the newline after <|im_end|> (end of the user turn)
    X = np.array(d["maps"]["Cc"]) / d["full"]["Cc"]
    return X[:, q] + X[:, nl]


def strip(d, v, label, sub, color):
    return dict(kind="exact", label=label, sub=sub, color=color, layers=d["layers"], v=[round(float(x), 3) for x in v])


def span(l0, l1, label, sub, ref, color="ink"):
    return dict(kind="prior", label=label, sub=sub, color=color, l0=l0, l1=l1, ref=ref)


panels = []
d = load("patch_4b")
panels.append(dict(name="Qwen3-VL-4B", tag="실패", D=d["D"], beh=d["behaviour"],
                   summary="번호를 꺼내 검색(24층)한 뒤에야 답 자리에서 j가 묶인다(32–34층). 검색은 묶이기 전의 k를 써서 표면 그림을 가져온다.",
                   line=dict(l=24, label="검색 24층 (§389)"),
                   rows=[strip(d, col(d, "Cc", lambda i, x: x.strip() == "third"), "“ third”", "서수 포인터 k", "ink"),
                         strip(d, col(d, "Cc", lambda i, x: x.strip() in ("image", "end")), "“ image”·“ end”", "지칭구로 퍼짐", "ink"),
                         strip(d, answer(d, "Cc"), "답 자리", "서수 정보 도착", "ink"),
                         span(20, 30, "답 자리", "k와 N이 따로 있음 (DAS)", "§470", "muted"),
                         span(32, 35, "답 자리", "j = N+1−k 로 묶임 (DAS)", "§470", "accent"),
                         strip(d, answer(d, "B"), "답 자리", "3번(표면) 그림 내용", "surface"),
                         strip(d, answer(d, "A"), "답 자리", "5번(정답) 그림 내용", "target")]))
d = load("patch2_35_9b")
panels.append(dict(name="Qwen3.5-9B", tag="부분", D=d["D"], beh=d["behaviour"],
                   summary="“?”에 서수 정보를 이른 층(11–17층)에 잠깐 싣지만, 정답과 표면 그림 내용이 같은 층(27층)에 함께 도착한다.",
                   line=dict(l=27, label="내용 도착 27층"),
                   rows=[strip(d, col(d, "Cc", lambda i, x: x.strip() == "third"), "“ third”", "서수 포인터 k", "ink"),
                         strip(d, relay(d), "“?”·턴 끝 줄바꿈", "이른 중계", "ink"),
                         strip(d, answer(d, "Cc"), "답 자리", "서수 정보 도착", "ink"),
                         strip(d, answer(d, "B"), "답 자리", "3번(표면) 그림 내용", "surface"),
                         strip(d, answer(d, "A"), "답 자리", "5번(정답) 그림 내용", "target")]))
d = load("patch_27b")
panels.append(dict(name="Qwen3.5-27B", tag="합성", D=d["D"], beh=d["behaviour"],
                   summary="서수 정보를 “?”·턴 끝 줄바꿈에 붙잡아 두고(27–47층), 끝을 읽고(47층) j를 만든 뒤(48–50층) 51층에서 검색한다.",
                   line=dict(l=51, label="검색 51층 (§597)"),
                   rows=[strip(d, col(d, "Cc", lambda i, x: x.strip() == "third"), "“ third”", "서수 포인터 k", "ink"),
                         strip(d, relay(d), "“?”·턴 끝 줄바꿈", "중계 · 붙잡기", "ink"),
                         span(47, 47, "질문 뒤쪽", "끝 N 읽기 (일반 주의)", "§597–§599", "muted"),
                         span(48, 50, "질문 뒤쪽", "j = N+1−k 합성", "§600", "accent"),
                         strip(d, answer(d, "Cc"), "답 자리", "서수 정보 도착", "ink"),
                         strip(d, answer(d, "A"), "답 자리", "5번(정답) 그림 내용", "target"),
                         strip(d, answer(d, "B"), "답 자리", "3번(표면) 그림 내용", "surface")]))
d = load("patch_4b_r3")
panels.append(dict(name="Qwen3-VL-4B + R³", tag="이식", D=d["D"], beh=d["behaviour"],
                   summary="R³가 1층에서 “ third”를 ‘다섯째’로, “ end”를 “ start”로 바꿔 써 둔다. 모델 자신의 검색(같은 23–25층)이 이번에는 정답 그림을 가져온다.",
                   line=dict(l=24, label="검색 (원래와 같은 층)"),
                   rows=[span(1, 1, "“ third”·“ end”", "R³ 쓰기: 합성 결과를 넣음", "R³", "accent"),
                         strip(d, col(d, "Cc", lambda i, x: x.strip() == "third"), "“ third”", "전방 포인터 ‘다섯째’", "ink"),
                         strip(d, col(d, "Cc", lambda i, x: x.strip() in ("image", "end")), "“ image”·“ end”", "퍼지지 않음", "ink"),
                         strip(d, answer(d, "Cc"), "답 자리", "서수 정보 도착", "ink"),
                         strip(d, answer(d, "A"), "답 자리", "5번(정답) 그림 내용", "target"),
                         dict(kind="none", label="답 자리", sub="3번(표면) 그림 내용", color="surface",
                              note="전체 효과 ≈ 0 (표면 그림을 바꿔도 답이 안 움직임)")]))
json.dump(panels, open(sys.argv[1], "w"), ensure_ascii=False)
print(f"-> {sys.argv[1]} 패널 {len(panels)}")
