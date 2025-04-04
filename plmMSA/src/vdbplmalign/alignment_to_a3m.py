def build_string(q, s):
    n = len(q)
    a3m = []
    for k in range(n):
        if q[k] == "-":
            a3m.append(s[k].lower())
        elif s[k] == "-":
            a3m.append("-")
        else:
            a3m.append(s[k])
    return "".join(a3m)


def parse_alignment_file(align_file):
    with open(align_file, "r") as f:
        lines = f.readlines()

    nline = len(lines)
    if not nline % 4 == 0:
        print(f"nline = {nline} % 4 not equal 0, check it")
        exit()

    head = lines[0]
    seq = lines[1].replace("-", "").upper()

    print(head.split()[0])
    print(seq[:-1])

    SEQS = set()
    A3M = {}
    heads = []
    scores = []

    for k in range(0, nline, 4):
        _, head, score = lines[k].split()
        a3m_string = build_string(lines[k + 1][:-1], lines[k + 3][:-1])
        if a3m_string in SEQS:
            print(f"{head} is already in SEQS")
            continue

        heads.append(head)
        scores.append(float(score))

        A3M[head] = a3m_string
        SEQS.add(a3m_string)

    return heads, scores, A3M


def filter_and_print_alignments(heads, scores, A3M, a3m_path, cutoff=None, ratio=0.2):
    order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    k = order[0]
    top_score = scores[k]
    head = heads[k]

    if cutoff is None:
        cutoff = top_score * ratio

    count = 0
    with open(a3m_path, "w") as f:
        for k in order:
            head = heads[k]
            score = scores[k]
            if float(score) < cutoff:
                continue
            f.write(f">{head} {score:8.3f}\n")
            f.write(A3M[head] + "\n")
            count += 1

        # f.write(
        #     f"highest score={top_score} cutoff={cutoff} count={count} head={head}\n"
        # )


def alignment_to_a3m(align_file, a3m_path, cutoff=None):
    heads, scores, A3M = parse_alignment_file(align_file)
    filter_and_print_alignments(heads, scores, A3M, a3m_path, cutoff)
